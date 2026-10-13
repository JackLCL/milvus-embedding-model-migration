"""Independently verify recorded Recall@10/nDCG@10 from qrels and retrieved IDs."""
import csv,json,math,statistics,sys
from pathlib import Path
root=Path(__file__).resolve().parent
run=root/'artifacts'/(root/'artifacts/LATEST_SUCCESS').read_text().strip()
qrels={}
for row in csv.DictReader((root/'artifacts/scifact/qrels/test.tsv').open(),delimiter='\t'):
    if int(row['score'])>0:qrels.setdefault(row['query-id'],{})[row['corpus-id']]=int(row['score'])
for version in ['A','B']:
    results=json.loads((run/f'retrieval_{version}.json').read_text())
    published=json.loads((run/f'metrics_{version}.json').read_text())
    assert set(results)==set(qrels)
    recalls=[];ndcgs=[]
    for qid,row in results.items():
        ranked=row['retrieved_ids'][:10];relevant=qrels[qid]
        assert len(ranked)==len(set(ranked))==10
        recalls.append(sum(doc in relevant for doc in ranked)/len(relevant))
        gains=[2**relevant.get(doc,0)-1 for doc in ranked]
        dcg=sum(gain/math.log2(i+2) for i,gain in enumerate(gains))
        ideal=sorted([2**rel-1 for rel in relevant.values()],reverse=True)[:10]
        idcg=sum(gain/math.log2(i+2) for i,gain in enumerate(ideal))
        ndcgs.append(dcg/idcg)
    actual={'recall_at_10':statistics.mean(recalls),'ndcg_at_10':statistics.mean(ndcgs)}
    for key,value in actual.items():assert abs(value-published[key])<1e-12
    print(version,actual)
print('Metrics independently checked against official test qrels.')
