const fs=require('fs'),path=require('path');
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'comparison-metrics.json'),'utf8'));
const C={ink:'#1F2329',muted:'#646A73',blue:'#5178C6',green:'#509863',line:'#BBBFC4',white:'#FFFFFF'};
let nodes=[],z=0;
function rect(id,x,y,w,h,fill){nodes.push({id,type:'composite_shape',x,y,width:w,height:h,z_index:z++,composite_shape:{type:'rect'},style:{fill_color:fill,fill_opacity:100,border_color:fill,border_style:'none'}})}
function text(id,x,y,w,s,size=20,bold=false,color=C.ink,align='left'){nodes.push({id,type:'text_shape',x,y,width:w,height:Math.ceil(size*1.8),z_index:100+z++,text:{text:s,font_size:size,font_weight:bold?'bold':'regular',text_color:color,horizontal_align:align,vertical_align:'mid'},style:{fill_opacity:0,border_style:'none'}})}
function line(id,x1,y1,x2,y2){nodes.push({id,type:'connector',x:Math.min(x1,x2),y:Math.min(y1,y2),width:Math.abs(x2-x1),height:Math.abs(y2-y1),z_index:40,style:{border_color:C.line,border_width:'narrow',border_style:'solid'},connector:{shape:'straight',start:{position:{x:x1,y:y1},arrow_style:'none'},end:{position:{x:x2,y:y2},arrow_style:'none'}}})}
rect('canvas',0,0,1120,574,C.white);
text('title',48,24,1024,'Embedding 模型替换前后的检索效果',32,true);
text('conditions',48,83,1024,'SciFact · 5,183 篇文献 · 300 条测试查询 · 指标越高越好',20,false,C.muted);
rect('legendA',48,142,20,20,C.blue);text('legendAt',80,130,420,'bge-small-en-v1.5 · 384 维',20,false,C.blue);
rect('legendB',584,142,20,20,C.green);text('legendBt',616,130,420,'bge-base-en-v1.5 · 768 维',20,false,C.green);
const panels=[['recall_at_10','Recall@10',48],['ndcg_at_10','nDCG@10',584]];
for(const [key,title,x] of panels){
 const m=data.metrics[key],barX=x+124,barW=274;
 text(key+'title',x,206,460,title,24,true);
 for(const [v,value,y,color] of [['A',m.before,272,C.blue],['B',m.after,340,C.green]]){
  text(key+v+'label',x,262+(y-272),116,v==='A'?'BGE-small':'BGE-base',20,true,color);
  rect(key+v+'bar',barX,y,barW*value,30,color);
  text(key+v+'value',barX+barW*value+10,y-8,100,value.toFixed(4),20,false,C.ink);
 }
 line(key+'axis',barX,408,barX+barW,408);
 for(const tick of [0,.5,1]){const tx=barX+tick*barW;line(key+'tick'+tick,tx,408,tx,415);text(key+'ticktext'+tick,tx-30,420,60,tick.toFixed(1),20,false,C.muted,'center');}
 text(key+'delta',x,482,472,'绝对增量 +'+m.absolute_delta.toFixed(4)+'  ｜  相对提升 '+m.relative_change_percent.toFixed(2)+'%',20,true,C.green);
}
text('foot',48,535,1024,'统一坐标范围 0—1；数值为同一测试集上逐查询指标的平均值。',20,false,C.muted);
fs.writeFileSync(path.join(__dirname,'comparison-native-nodes.json'),JSON.stringify({nodes},null,2));
console.log(nodes.length+' editable chart nodes');
// A compact SVG is also provided for publication outside Feishu.
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
let svg='<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="574" viewBox="0 0 1120 574"><style>text{font-family:"Noto Sans CJK SC","PingFang SC",sans-serif}</style>';
for(const n of [...nodes].sort((a,b)=>a.z_index-b.z_index)){
 if(n.type==='composite_shape')svg+=`<rect x="${n.x}" y="${n.y}" width="${n.width}" height="${n.height}" fill="${n.style.fill_color}"/>`;
 if(n.type==='connector'){const a=n.connector.start.position,b=n.connector.end.position;svg+=`<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${C.line}" stroke-width="1.5"/>`;}
 if(n.type==='text_shape'){const t=n.text,x=t.horizontal_align==='center'?n.x+n.width/2:n.x;svg+=`<text x="${x}" y="${n.y+n.height/2+t.font_size*.35}" text-anchor="${t.horizontal_align==='center'?'middle':'start'}" font-size="${t.font_size}" font-weight="${t.font_weight==='bold'?700:400}" fill="${t.text_color}">${escape(t.text)}</text>`;}
}
svg+='</svg>';fs.writeFileSync(path.join(__dirname,'comparison.svg'),svg);
