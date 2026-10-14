#!/usr/bin/env python3
"""Reproducible full-corpus SciFact embedding migration + real HTTP inference.
Run: .venv/bin/python run_public.py --uri http://localhost:19530
"""
import argparse, csv, hashlib, json, math, os, platform, shutil, statistics, threading, time, traceback, urllib.request, zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
import torch
import importlib.metadata
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from pymilvus import MilvusClient, DataType

ROOT=Path(__file__).resolve().parent
MODELS={'A':'BAAI/bge-small-en-v1.5','B':'BAAI/bge-base-en-v1.5'}
FIELDS={'A':'vector_a','B':'vector_b'}
DIMS={'A':384,'B':768}
MODEL_REVISIONS={'A': '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a', 'B': 'a5beb1e3e68b9ab74eb54cfd186867f64f240e1a'}
WORKERS=4
TORCH_THREADS=2
MODEL_BATCH_SIZE=8
DATA_URL='https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
PREFIX='Represent this sentence for searching relevant passages: '

def save(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')

def load_data():
    archive=ROOT/'artifacts/scifact.zip'
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        with urllib.request.urlopen(DATA_URL,timeout=120) as response:
            archive.write_bytes(response.read())
    if not (archive.parent/'scifact/corpus.jsonl').exists():
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.startswith('/') or '..' in Path(name).parts: raise ValueError(name)
            z.extractall(archive.parent)
    d=archive.parent/'scifact'
    docs=[json.loads(line) for line in (d/'corpus.jsonl').read_text().splitlines()]
    queries={r['_id']:r['text'] for r in map(json.loads,(d/'queries.jsonl').read_text().splitlines())}
    qrels=defaultdict(dict)
    for row in csv.DictReader((d/'qrels/test.tsv').open(),delimiter='\t'):
        if int(row['score'])>0:qrels[row['query-id']][row['corpus-id']]=int(row['score'])
    return docs,queries,dict(qrels)

class Lab:
    def __init__(self,uri,port,cache):
        self.uri,self.port,self.cache=uri,port,cache
        self.run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.out=ROOT/'artifacts'/self.run;self.out.mkdir(parents=True)
        shutil.copyfile(__file__,self.out/'run_public.py')
        self.name='scifact_switch_'+self.run.lower()
        self.c=MilvusClient(uri=uri,timeout=60)
        self.checks={};self.stage='prepare';self.route='A';self.ready={'A':False,'B':False}
        self.route_lock=threading.Lock();self.locks={v:[threading.Lock() for _ in range(WORKERS)] for v in MODELS}
        self.pool=ThreadPoolExecutor(max_workers=WORKERS)
        self.stop=threading.Event();self.samples=[];self.reader_thread=None;self.server=None
        self.start=time.monotonic();self.counts={};self.expected={v:{} for v in MODELS}
        self.events=(self.out/'events.jsonl').open('a',buffering=1)
    def event(self,event,**kw):
        r={'elapsed_s':round(time.monotonic()-self.start,3),'event':event,**kw}
        line=json.dumps(r,ensure_ascii=False,default=str);print(line,flush=True);self.events.write(line+'\n')
    def check(self,name,passed,**kw):
        self.checks[name]={'passed':bool(passed),**kw};self.event('check',name=name,**self.checks[name])
        if not passed:raise AssertionError(name)
    def phase(self,name):self.stage=name;self.event('stage',name=name)
    def encode(self,v,texts,query=False):
        # Query instructions are applied once by encode_queries().
        values=list(texts)
        def shard(i,chunk):
            with self.locks[v][i]:
                model=self.models[v][i]
                vectors=model.encode_queries(chunk) if query else model.encode_documents(chunk)
                return [x.tolist() for x in vectors]
        if len(values)<WORKERS:return shard(0,values)
        size=math.ceil(len(values)/WORKERS)
        futures=[self.pool.submit(shard,i,values[i*size:(i+1)*size]) for i in range(WORKERS) if values[i*size:(i+1)*size]]
        return [x for f in futures for x in f.result()]
    def index(self,field):
        p=self.c.prepare_index_params()
        p.add_index(field_name=field,index_name=field+'_hnsw',index_type='HNSW',metric_type='COSINE',params={'M':16,'efConstruction':100})
        self.c.create_index(self.name,index_params=p,timeout=180)
    def search(self,text,version=None):
        with self.route_lock:
            v=version or self.route
            if v not in self.ready or not self.ready[v]:raise ValueError('retrieval version is not ready')
        # Pin v BEFORE inference; never read the global route again in this request.
        vector=self.encode(v,[text],query=True)[0]
        hits=self.c.search(self.name,data=[vector],anns_field=FIELDS[v],limit=10,
            output_fields=['title'],search_params={'metric_type':'COSINE','params':{'ef':128}},
            consistency_level='Strong',timeout=10)[0]
        return {'version':v,'model':MODELS[v],'field':FIELDS[v],'dimension':len(vector),'hits':hits}
    def start_http(self):
        lab=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                try:
                    size=int(self.headers.get('Content-Length','0'))
                    if size>16384:raise ValueError('request too large')
                    body=json.loads(self.rfile.read(size))
                    if self.path=='/search': result=lab.search(body['query'],body.get('version'))
                    elif self.path=='/route':
                        v=body['version']
                        with lab.route_lock:
                            if not lab.ready.get(v):raise ValueError('version is not ready')
                            lab.route=v
                        result={'active_version':v}
                    else:raise ValueError('unknown path')
                    status=200
                except Exception as exc:status=400;result={'error':str(exc)}
                data=json.dumps(result).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        self.server=ThreadingHTTPServer(('127.0.0.1',self.port),Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
    def post(self,path,body):
        req=urllib.request.Request(f'http://127.0.0.1:{self.port}'+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)
    def monitor(self):
        i=0
        with (self.out/'http_requests.jsonl').open('a',buffering=1) as out:
            while not self.stop.is_set():
                start=time.monotonic();sample={'stage':self.stage,'start_s':start-self.start}
                try:
                    result=self.post('/search',{'query':self.queries[self.qids[i%len(self.qids)]]})
                    sample.update(ok=len(result['hits'])==10,version=result['version'],dimension=result['dimension'],field=result['field'])
                except Exception as exc:sample.update(ok=False,error=str(exc))
                sample['latency_ms']=(time.monotonic()-start)*1000
                self.samples.append(sample);out.write(json.dumps(sample)+'\n');i+=1;self.stop.wait(.25)
    def evaluate(self,v):
        vectors=self.encode(v,[self.queries[q] for q in self.qids],query=True)
        results={};recalls=[];ndcgs=[]
        for begin in range(0,len(vectors),32):
            hits=self.c.search(self.name,data=vectors[begin:begin+32],anns_field=FIELDS[v],limit=10,
                search_params={'metric_type':'COSINE','params':{'ef':128}},consistency_level='Strong',timeout=30)
            for qid,items in zip(self.qids[begin:begin+32],hits):
                ids=[str(h['id']) for h in items];rel=self.qrels[qid]
                recalls.append(len(set(ids)&set(rel))/len(rel))
                dcg=sum((2**rel.get(pk,0)-1)/math.log2(rank+2) for rank,pk in enumerate(ids))
                idcg=sum((2**score-1)/math.log2(rank+2) for rank,score in enumerate(sorted(rel.values(),reverse=True)[:10]))
                ndcgs.append(dcg/idcg if idcg else 0)
                results[qid]={'query':self.queries[qid],'relevant_ids':rel,'retrieved_ids':ids,'recall_at_10':recalls[-1],'ndcg_at_10':ndcgs[-1]}
        metrics={'queries':len(self.qids),'recall_at_10':statistics.mean(recalls),'ndcg_at_10':statistics.mean(ndcgs)}
        save(self.out/f'retrieval_{v}.json',results);save(self.out/f'metrics_{v}.json',metrics)
        self.event('evaluation',version=v,**metrics);return metrics
    def save_model_provenance(self):
        packages={p:importlib.metadata.version(p) for p in
                  ['pymilvus','pymilvus.model','sentence-transformers','transformers','torch']}
        models=[]
        for v,model_name in MODELS.items():
            snapshot=Path(self.cache)/('models--'+model_name.replace('/','--'))/'snapshots'/MODEL_REVISIONS[v]
            files=[]
            for path in sorted(snapshot.rglob('*')):
                if path.is_file():
                    files.append({'file':path.relative_to(snapshot).as_posix(),
                                  'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
                                  'bytes':path.stat().st_size})
            models.append({'version':v,'repository':model_name,'revision':MODEL_REVISIONS[v],
                           'dimension':DIMS[v],'dtype':str(next(self.models[v][0].model.parameters()).dtype),
                           'max_seq_length':512,'pooling':self.models[v][0].model[1].get_config_dict(),'files':files})
        save(self.out/'model-provenance.json',{
            'python':platform.python_version(),'packages':packages,
            'embedding_library':'pymilvus.model.SentenceTransformerEmbeddingFunction',
            'backend':'PyTorch','device':'cpu','quantized':False,
            'workers':WORKERS,'torch_threads':TORCH_THREADS,'model_batch_size':MODEL_BATCH_SIZE,
            'normalize_embeddings':True,'query_instruction':PREFIX,'doc_instruction':'','models':models,
        })
    def run_all(self):
        self.check('server_is_3_0_2',self.c.get_server_version()=='3.0.2')
        docs,self.queries,self.qrels=load_data();self.qids=sorted(self.qrels,key=int)
        self.check('full_scifact_test',len(docs)==5183 and len(self.qids)==300,documents=len(docs),queries=len(self.qids))
        self.source={r['_id']:{'id':r['_id'],'title':r['title'],'text':r['title']+'\n'+r['text']} for r in docs}
        save(self.out/'provenance.json',{'dataset':DATA_URL,'dataset_sha256':hashlib.file_digest((ROOT/'artifacts/scifact.zip').open('rb'),'sha256').hexdigest(),
            'collection':self.name,'uri':self.uri,'models':MODELS,'model_revisions':MODEL_REVISIONS,'embedding_library':'pymilvus.model','dimensions':DIMS,'query_prefix':PREFIX,'python':platform.python_version(),
            'documents':len(docs),'queries':len(self.qids),'text_policy':'title + newline + abstract; one document per entity; model truncates at 512 tokens; no second chunking'})
        torch.set_num_threads(TORCH_THREADS)
        torch.set_num_interop_threads(1)
        self.models={}
        for v,model_name in MODELS.items():
            self.models[v]=[]
            for _ in range(WORKERS):
                model=SentenceTransformerEmbeddingFunction(
                    model_name=model_name,device='cpu',batch_size=MODEL_BATCH_SIZE,
                    query_instruction=PREFIX,doc_instruction='',normalize_embeddings=True,
                    cache_folder=self.cache,revision=MODEL_REVISIONS[v],
                )
                model.model.max_seq_length=512
                assert model.dim==DIMS[v]
                self.models[v].append(model)
        self.save_model_provenance()
        schema=self.c.create_schema(auto_id=False,enable_dynamic_field=False)
        schema.add_field('id',DataType.VARCHAR,max_length=64,is_primary=True)
        schema.add_field('title',DataType.VARCHAR,max_length=2048)
        schema.add_field('text',DataType.VARCHAR,max_length=65535)
        schema.add_field('vector_a',DataType.FLOAT_VECTOR,dim=384)
        self.c.create_collection(self.name,schema=schema,consistency_level='Strong')
        self.phase('ingest_A')
        values=list(self.source.values())
        for begin in range(0,len(values),32):
            batch=values[begin:begin+32];vectors=self.encode('A',[r['text'] for r in batch])
            self.c.insert(self.name,data=[{**r,'vector_a':vec} for r,vec in zip(batch,vectors)])
            self.expected['A'].update({r['id']:vec for r,vec in zip(batch,vectors)})
            if begin%320==0:self.event('ingested',rows=min(begin+32,len(values)))
        self.c.flush(self.name,timeout=180);self.index('vector_a');self.c.load_collection(self.name,timeout=180)
        save(self.out/'schema_A.json',self.c.describe_collection(self.name));self.ready['A']=True
        self.metric_a=self.evaluate('A')
        self.start_http();self.phase('baseline_A')
        self.reader_thread=threading.Thread(target=self.monitor,daemon=True);self.reader_thread.start();time.sleep(5)
        example_q=self.queries[self.qids[0]]
        save(self.out/'http_example_A.json',self.post('/search',{'query':example_q}))
        self.phase('add_B')
        self.c.add_collection_field(self.name,field_name='vector_b',data_type=DataType.FLOAT_VECTOR,dim=768,nullable=True)
        self.index('vector_b');self.c.refresh_load(self.name,timeout=180)
        self.check('old_and_new_dimensions', {f['name']:f.get('params',{}).get('dim') for f in self.c.describe_collection(self.name)['fields'] if f['name'].startswith('vector_')}=={'vector_a':384,'vector_b':768})
        self.phase('backfill_B_from_milvus_text')
        it=self.c.query_iterator(self.name,filter='',output_fields=['id','text'],batch_size=32,consistency_level='Strong')
        processed=0
        try:
            while True:
                batch=it.next()
                if not batch:break
                self.check_text_batch(batch)
                vectors=self.encode('B',[row['text'] for row in batch])
                self.c.upsert(self.name,data=[{'id':row['id'],'vector_b':vec} for row,vec in zip(batch,vectors)],partial_update=True)
                self.expected['B'].update({r['id']:vec for r,vec in zip(batch,vectors)})
                processed+=len(batch)
                if processed==32:
                    self.c.flush(self.name,timeout=180)
                    self.c.load_collection(self.name,load_fields=['id','title','text','vector_a','vector_b'],timeout=180)
                    self.c.refresh_load(self.name,timeout=180)
                    ids={r['id'] for r in batch}
                    hits=self.c.search(self.name,data=[vectors[0]],anns_field='vector_b',limit=100,consistency_level='Strong')[0]
                    self.check('NULL_B_rows_excluded',len(hits)==32 and all(h['id'] in ids for h in hits))
                if processed%320==0 or processed==len(values):self.event('backfilled_from_milvus',rows=processed)
        finally:it.close()
        self.check('all_documents_backfilled',processed==len(values)==len(self.expected['B']))
        self.phase('audit_and_evaluate_B');self.c.flush(self.name,timeout=180);self.c.refresh_load(self.name,timeout=180)
        deadline=time.monotonic()+180
        while True:
            desc=self.c.describe_index(self.name,'vector_b_hnsw')
            if desc.get('state')=='Finished' and desc.get('pending_index_rows',0)==0:break
            if time.monotonic()>deadline:raise TimeoutError(str(desc))
            time.sleep(1)
        self.event('B_index_ready',description=desc)
        audit=self.c.query_iterator(self.name,filter='',output_fields=['id','title','text','vector_a','vector_b'],batch_size=64,consistency_level='Strong')
        seen=set();err={'A':0.,'B':0.}
        try:
            while True:
                batch=audit.next()
                if not batch:break
                for row in batch:
                    pk=row['id'];assert pk not in seen;seen.add(pk)
                    assert row['text']==self.source[pk]['text'] and row['title']==self.source[pk]['title']
                    for v in MODELS:err[v]=max(err[v],float(np.max(np.abs(np.asarray(row[FIELDS[v]])-np.asarray(self.expected[v][pk])))))
        finally:audit.close()
        self.check('all_texts_and_vectors_audited',seen==set(self.source) and max(err.values())<1e-6,rows=len(seen),max_error=err)
        self.metric_b=self.evaluate('B');self.ready['B']=True
        save(self.out/'schema_AB.json',self.c.describe_collection(self.name))
        try:
            self.c.search(self.name,data=[self.expected['A'][next(iter(self.source))]],anns_field='vector_b',limit=1,timeout=10)
        except Exception as exc:self.check('wrong_dimension_rejected','dim' in str(exc).lower(),message=str(exc))
        else:self.check('wrong_dimension_rejected',False)
        self.phase('canary_AB');canary=[]
        for i in range(30):
            v='B' if i%5==0 else 'A'
            r=self.post('/search',{'query':self.queries[self.qids[i]],'version':v})
            assert r['version']==v and r['dimension']==DIMS[v] and len(r['hits'])==10
            canary.append({'version':v,'query_id':self.qids[i]})
        save(self.out/'canary.json',canary);self.check('canary_routing',len(canary)==30,B_requests=sum(r['version']=='B' for r in canary))
        self.phase('cutover_B');self.post('/route',{'version':'B'});time.sleep(5)
        save(self.out/'http_example_B.json',self.post('/search',{'query':example_q}))
        # Isolated extra document: after evaluation, so it cannot contaminate metrics.
        probe={'id':'migration_probe','title':'Embedding migration probe','text':'The model migration keeps both retrieval versions updated during the rollback window.'}
        for rev in [1,2]:
            probe['text']=probe['text']+f' Revision {rev}.'
            self.c.upsert(self.name,data=[{**probe,**{FIELDS[v]:self.encode(v,[probe['text']])[0] for v in MODELS}}])
        self.phase('rollback_A');self.post('/route',{'version':'A'});time.sleep(5)
        for v in MODELS:
            r=self.post('/search',{'query':probe['text'],'version':v})
            self.check(v+'_latest_update_searchable',any(h['id']==probe['id'] for h in r['hits']))
        self.c.delete(self.name,ids=[probe['id']]);self.check('probe_deleted',not self.c.get(self.name,ids=[probe['id']],consistency_level='Strong'))
        self.phase('final_B');self.post('/route',{'version':'B'});time.sleep(6)
        # This low-concurrency lab uses a fixed observation interval.
        # Production deployments must explicitly drain old-version requests.
        self.ready['A']=False
        self.phase('drop_A');self.c.drop_collection_field(self.name,field_name='vector_a',timeout=120)
        final=self.c.describe_collection(self.name);save(self.out/'schema_B.json',final)
        self.check('old_field_and_index_removed','vector_a' not in [f['name'] for f in final['fields']] and 'vector_a_hnsw' not in self.c.list_indexes(self.name))
        probe['id']='after_cleanup';self.c.insert(self.name,data=[{**probe,'vector_b':self.encode('B',[probe['text']])[0]}])
        self.check('B_only_write',bool(self.c.get(self.name,ids=[probe['id']],consistency_level='Strong')))
        self.c.delete(self.name,ids=[probe['id']]);self.phase('after_cleanup');time.sleep(5)
        self.check('final_count',self.c.query(self.name,filter='',output_fields=['count(*)'],consistency_level='Strong')[0]['count(*)']==5183)
        after=self.evaluate('B');self.check('cleanup_preserves_retrieval',after==self.metric_b)
    def check_text_batch(self,batch):
        for row in batch:assert row['text']==self.source[row['id']]['text']
    def finish(self,error):
        self.stop.set()
        if self.reader_thread:self.reader_thread.join(timeout=40)
        if self.server:self.server.shutdown();self.server.server_close()
        self.checks['continuous_real_inference_HTTP']={'passed':bool(self.samples) and all(s['ok'] for s in self.samples),'requests':len(self.samples),'failures':sum(not r['ok'] for r in self.samples)}
        groups=defaultdict(list)
        for row in self.samples:groups[row['stage']].append(row)
        stats={k:{'requests':len(rows),'failures':sum(not r['ok'] for r in rows),'p95_ms':float(np.percentile([r['latency_ms'] for r in rows],95))} for k,rows in groups.items()}
        result={'success':not error and all(c['passed'] for c in self.checks.values()),'error':error,'collection':self.name,'elapsed_s':time.monotonic()-self.start,'checks':self.checks,'http_by_stage':stats,'metrics_A':getattr(self,'metric_a',None),'metrics_B':getattr(self,'metric_b',None)}
        save(self.out/'summary.json',result);self.event('finished',success=result['success'],output=str(self.out))
        if result['success']:(ROOT/'artifacts/LATEST_SUCCESS').write_text(self.run+'\n')
        self.pool.shutdown();self.c.close();self.events.close();return result['success']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--uri',default='http://localhost:19530');p.add_argument('--port',type=int,default=8080);p.add_argument('--model-cache',default=str(ROOT/'models'));args=p.parse_args()
    lab=Lab(args.uri,args.port,args.model_cache);error=None
    try:lab.run_all()
    except Exception:error=traceback.format_exc();print(error,flush=True)
    raise SystemExit(0 if lab.finish(error) else 1)
