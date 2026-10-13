const fs=require('fs'),path=require('path');
const C={ink:'#1F2329',muted:'#646A73',blue:'#5178C6',blueBg:'#F0F4FC',green:'#509863',greenBg:'#DFF5E5',amber:'#D4B45B',amberBg:'#FEF1CE',red:'#D25D5A',redBg:'#FEE3E2',line:'#BBBFC4',white:'#FFFFFF'};
let nodes=[],z=0;
function shape(id,x,y,w,h,fill=C.white,border=C.line,type='round_rect',label='',size=20){
 const n={id,type:'composite_shape',x,y,width:w,height:h,z_index:z++,composite_shape:{type},style:{fill_color:fill,fill_opacity:100,border_color:border,border_width:'narrow',border_style:'solid'}};
 if(label)n.text={text:label,font_size:size,horizontal_align:'center',vertical_align:'mid',text_color:C.ink};nodes.push(n);return n;
}
function text(id,x,y,w,s,size=20,bold=false,color=C.ink,align='left'){
 const n={id,type:'text_shape',x,y,width:w,height:Math.ceil(s.split('\n').length*size*1.8),z_index:100+z++,style:{fill_opacity:0,border_style:'none'},text:{text:s,font_size:size,font_weight:bold?'bold':'regular',horizontal_align:align,vertical_align:'mid',text_color:color}};nodes.push(n);return n;
}
function table(id,x,y,cols,rows,styles={},rh=36){
 const cells=[];
 rows.forEach((row,r)=>row.forEach((s,c)=>{
  const custom=styles[`${r},${c}`]||styles[r]||{};
  cells.push({row_index:r+1,col_index:c+1,text:{text:s,font_size:18,font_weight:r===0?'bold':'regular',horizontal_align:'center',vertical_align:'mid',text_color:custom.color||C.ink,line_through:!!custom.strike},style:{fill_color:custom.fill||(r===0?C.blueBg:C.white),fill_color_type:1,fill_opacity:100}});
 }));
 const n={id,type:'table',x,y,width:cols.reduce((a,b)=>a+b),height:rh*rows.length,z_index:50+z++,table:{meta:{row_sizes:rows.map(()=>rh),col_sizes:cols,style:{fill_color:C.white,fill_color_type:1,fill_opacity:100,border_color:C.line,border_color_type:1,border_style:'solid',border_width:'narrow'},text:{font_size:18,text_color:C.ink,horizontal_align:'center',vertical_align:'mid'}},cells}};nodes.push(n);return n;
}
function pos(id,side){const n=nodes.find(n=>n.id===id);return side==='right'?[n.x+n.width,n.y+n.height/2]:side==='left'?[n.x,n.y+n.height/2]:side==='bottom'?[n.x+n.width/2,n.y+n.height]:[n.x+n.width/2,n.y];}
function line(id,from,to,fa='right',ta='left',dash=false,points=[]){
 const n={id,type:'connector',x:0,y:0,width:0,height:0,z_index:40,style:{border_color:C.line,border_width:'narrow',border_style:dash?'dash':'solid'},connector:{shape:'polyline',start:{attached_object:{id:from,snap_to:fa},arrow_style:'none'},end:{attached_object:{id:to,snap_to:ta},arrow_style:'triangle_arrow'}}};
 if(points.length){const a=pos(from,fa),b=pos(to,ta),bx=Math.min(a[0],b[0]),by=Math.min(a[1],b[1]);n.connector.turning_points=points.map(p=>({x:p.x-bx,y:p.y-by}));n.preview_points=points;}nodes.push(n);
}
shape('canvas',0,0,1232,1176,C.white,C.white,'rect');
text('title',64,24,1104,'Switch embedding models | Field and data changes',32,true);
const legend=text('legend',64,82,1104,'Blue: BGE-small (bge-small-en-v1.5)    Green: BGE-base (bge-base-en-v1.5)\n✕ Remove field    ✓ Vector ready',20,false,C.muted);
const legendRun=(text,color)=>({element_type:0,text_element:{text,text_style:{font_size:20,font_weight:'regular',text_color:color}}});
legend.text.rich_text={paragraphs:[
 {paragraph_type:0,text_color_type:1,elements:[legendRun('Blue: BGE-small (bge-small-en-v1.5)    ',C.blue),legendRun('Green: BGE-base (bge-base-en-v1.5)',C.green)]},
 {paragraph_type:0,text_color_type:1,elements:[legendRun('✕ Remove field    ',C.red),legendRun('✓ Vector ready',C.green)]},
]};
const xs=[64,448,832],y1=172,y2=634,W=336,H=350;
const titles=['Serve with old model','Add new vector field','Re-embed documents','Check data and quality','Shift traffic with rollback','Drop the old vector field'];
for(let i=0;i<6;i++){const x=xs[i%3],y=i<3?y1:y2;shape('s'+(i+1),x,y,W,H,C.white,i<3?C.blue:i===3?C.amber:C.green);text('title'+(i+1),x+18,y+18,W-36,`0${i+1}  ${titles[i]}`,22,true);}
// 1: source document -> encoder -> database, followed by the initial schema.
shape('doc',80,242,56,50,C.white,C.blue,'document_shape','Text',18);
shape('encoderA',164,242,106,50,C.blueBg,C.blue,'round_rect','BGE-small',18);
shape('db',300,228,80,80,C.white,C.blue,'cylinder');text('dbLabel',300,264,80,'Milvus',18,false,C.ink,'center');
line('doc-a','doc','encoderA');line('a-db','encoderA','db');
table('schemaA',84,318,[182,114],[['Field','Content'],['id','Key'],['title','Title'],['text','Text'],['vector_a','384 dims']],{4:{fill:C.blueBg,color:C.blue}},38);
// 2: genuine editable table; highlighted added field is a schema row, not an entity.
table('schemaAdded',468,246,[174,122],[['Field','State'],['id','Keep'],['title','Keep'],['text','Keep'],['vector_a','384 dims'],['+ vector_b','768 · NULL']],{4:{fill:C.blueBg,color:C.blue},5:{fill:C.greenBg,color:C.green}},40);
// 3: before/after entity samples. A is retained while B is filled.
table('beforeRows',850,246,[78,109,109],[['id','vector_a','vector_b'],['r1','✓','NULL'],['r2','✓','NULL']],{'1,1':{fill:C.blueBg,color:C.blue},'2,1':{fill:C.blueBg,color:C.blue}},42);
text('encodeB',850,382,296,'↓  BGE-base encoding  ↓',20,true,C.green,'center');
table('afterRows',850,430,[78,109,109],[['id','vector_a','vector_b'],['r1','✓','✓'],['r2','✓','✓']],{'1,1':{fill:C.blueBg,color:C.blue},'2,1':{fill:C.blueBg,color:C.blue},'1,2':{fill:C.greenBg,color:C.green},'2,2':{fill:C.greenBg,color:C.green}},42);
// 4: explicit checks plus the observed Recall@10 values, drawn to the same scale.
text('checks',84,700,296,'✓ Data   ✓ Index   ✓ Quality',18,false,C.green,'center');
table('metrics',84,782,[148,148],[['Model','Recall@10'],['BGE-small','0.8362'],['BGE-base','0.8742']],{1:{fill:C.blueBg,color:C.blue},2:{fill:C.greenBg,color:C.green}},48);
// 5: a query selects a complete model + field pair; old route remains for rollback.
shape('query',470,800,64,58,C.white,C.line,'bubble','Query',17);
shape('routeA',552,726,106,54,C.blueBg,C.blue,'round_rect','BGE-small',18);
shape('fieldA',670,720,96,66,C.blueBg,C.blue,'round_rect','vector_a\n384 dims',18);
shape('routeB',552,872,106,54,C.greenBg,C.green,'round_rect','BGE-base',18);
shape('fieldB',670,866,96,66,C.greenBg,C.green,'round_rect','vector_b\n768 dims',18);
line('query-a','query','routeA','right','left',true);line('query-b','query','routeB');line('route-a','routeA','fieldA','right','left',true);line('route-b','routeB','fieldB');
// 6: only the old vector field is crossed out; all document fields remain.
table('schemaDropped',852,708,[174,122],[['Field','State'],['id','Keep'],['title','Keep'],['text','Keep'],['vector_a','✕ Drop'],['vector_b','✓ Keep']],{4:{fill:C.redBg,color:C.red},'4,0':{fill:C.redBg,color:C.red,strike:true},5:{fill:C.greenBg,color:C.green}},40);
// Main six-stage flow plus the two required safety branches.
line('p12','s1','s2');line('p23','s2','s3');
line('p34','s3','s4','bottom','left',false,[{x:1000,y:576},{x:24,y:576},{x:24,y:809}]);
line('p45','s4','s5');line('p56','s5','s6');
text('passed',392,755,64,'Pass',20,false,C.muted,'center');text('stable',776,755,64,'Stable',16,false,C.muted,'center');
shape('hold',64,1060,336,54,C.redBg,C.red,'round_rect','Checks failed → keep BGE-small',18);
shape('rollback',448,1060,336,54,C.redBg,C.red,'round_rect','Regression → use BGE-small',18);
line('p4hold','s4','hold','bottom','top',true);line('p5rollback','s5','rollback','bottom','top',true);
// Reserve sufficient height for native table cell padding and text layout.
for(const n of nodes){
 if(/^s[1-6]$/.test(n.id))n.height+=(/^s[1-3]$/.test(n.id)?60:26);
 if(n.type!=='connector'){
  if(n.y>=1030)n.y+=52;
  else if(n.y>=634)n.y+=26;
 }
 if(n.id==='canvas')n.height=1240;
}
const bridge=nodes.find(n=>n.id==='p34');
bridge.preview_points=[{x:1000,y:616},{x:24,y:616},{x:24,y:848}];
const ba=pos('s3','bottom'),bb=pos('s4','left'),bx=Math.min(ba[0],bb[0]),by=Math.min(ba[1],bb[1]);
bridge.connector.turning_points=bridge.preview_points.map(p=>({x:p.x-bx,y:p.y-by}));
const clean=nodes.map(n=>{const o=JSON.parse(JSON.stringify(n));delete o.preview_points;return o;});
fs.writeFileSync(path.join(__dirname,'native-overview-nodes.json'),JSON.stringify({nodes:clean},null,2));
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
let svg='<svg xmlns="http://www.w3.org/2000/svg" width="1232" height="1240" viewBox="0 0 1232 1240"><style>text{font-family:Arial,Helvetica,sans-serif}</style><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#BBBFC4"/></marker></defs><rect width="1232" height="1240" fill="white"/>';
function label(x,y,w,h,t){
 if(t.rich_text){
  let yy=y+24;
  for(const p of t.rich_text.paragraphs){svg+=`<text x="${x+8}" y="${yy}" font-size="20" xml:space="preserve">`;for(const e of p.elements){const r=e.text_element;svg+=`<tspan fill="${r.text_style.text_color}">${esc(r.text)}</tspan>`;}svg+='</text>';yy+=32;}
  return;
 }
const align=t.horizontal_align||'center';const ax=align==='left'?x+8:align==='right'?x+w-8:x+w/2;const lines=(t.text||'').split('\n'),size=t.font_size||20;const lineh=size*1.4;let yy=y+(h-lines.length*lineh)/2+size*1.1;for(const l of lines){svg+=`<text x="${ax}" y="${yy}" text-anchor="${align==='left'?'start':align==='right'?'end':'middle'}" font-size="${size}" font-weight="${t.font_weight==='bold'?700:400}" fill="${t.text_color||C.ink}" ${t.line_through?'text-decoration="line-through"':''}>${esc(l)}</text>`;yy+=lineh;}}
for(const n of [...nodes].sort((a,b)=>a.z_index-b.z_index)){
 if(n.type==='connector'){const c=n.connector,a=pos(c.start.attached_object.id,c.start.attached_object.snap_to),b=pos(c.end.attached_object.id,c.end.attached_object.snap_to),pts=[a,...(n.preview_points||[]).map(p=>[p.x,p.y]),b];svg+=`<polyline points="${pts.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${C.line}" stroke-width="2" ${n.style.border_style==='dash'?'stroke-dasharray="6 5"':''} marker-end="url(#arr)"/>`;continue;}
 if(n.type==='composite_shape'){
  const {x,y,width:w,height:h}=n, fill=n.style.fill_color, stroke=n.style.border_color;
  if(n.composite_shape.type==='cylinder'){
   const ry=12;
   svg+=`<path d="M${x} ${y+ry} A${w/2} ${ry} 0 0 1 ${x+w} ${y+ry} L${x+w} ${y+h-ry} A${w/2} ${ry} 0 0 1 ${x} ${y+h-ry} Z" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/><ellipse cx="${x+w/2}" cy="${y+ry}" rx="${w/2}" ry="${ry}" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`;
  }else if(n.composite_shape.type==='document_shape'){
   svg+=`<path d="M${x} ${y} H${x+w} V${y+h-7} Q${x+w*.75} ${y+h-17} ${x+w/2} ${y+h-7} Q${x+w*.25} ${y+h+3} ${x} ${y+h-7} Z" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`;
  }else{
   svg+=`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`;
  }
  if(n.text)label(n.x,n.y,n.width,n.height,n.text);
 }
 if(n.type==='text_shape')label(n.x,n.y,n.width,n.height,n.text);
 if(n.type==='table'){let yy=n.y;for(let r=0;r<n.table.meta.row_sizes.length;r++){let xx=n.x;for(let c=0;c<n.table.meta.col_sizes.length;c++){const cell=n.table.cells.find(a=>a.row_index===r+1&&a.col_index===c+1),w=n.table.meta.col_sizes[c],h=n.table.meta.row_sizes[r];svg+=`<rect x="${xx}" y="${yy}" width="${w}" height="${h}" fill="${cell.style.fill_color}" stroke="${C.line}" stroke-width="1"/>`;label(xx,yy,w,h,cell.text);xx+=w;}yy+=n.table.meta.row_sizes[r];}}
}
svg+='</svg>';fs.writeFileSync(path.join(__dirname,'overview.svg'),svg);
console.log(clean.length+' native nodes; '+clean.filter(n=>n.type==='table').length+' editable tables');
