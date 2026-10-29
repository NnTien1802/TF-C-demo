/* GestureLab V4: vanilla JavaScript UI + real FastAPI TF-C inference.
   No Streamlit. No placeholder probabilities. FFT is a descriptive preview only. */
'use strict';

const $ = id => document.getElementById(id);
const state = {source:'example', file:null, data:null, window:0, busy:false, lastSignature:'', modelFound:false};
const COLORS = ['#5c86ed','#20b49b','#e68d6b'];
const fmt = (v,d=2) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : '—';
function txt(id,v){$(id).textContent=String(v);}
function showError(message=''){$('inlineError').hidden=!message;txt('inlineError',message);}
function setBusy(busy, predicting=false){
  state.busy=busy;
  $('predictButton').disabled=busy;
  txt('predictButtonText',busy?(predicting?'Đang chạy mô hình...':'Đang xử lý dữ liệu...'):'Run TF-C Prediction');
}
function setResultStatus(title,message,kind=''){
  const box=$('resultStatus');box.className='result-status'+(kind?' '+kind:'');
  txt('resultStatusIcon',kind==='success'?'✓':kind==='warning'?'!':kind==='error'?'×':'○');
  txt('resultStatusTitle',title);txt('resultStatusText',message);
}
function chooseSource(source){
  if(state.source===source)return;
  state.source=source;
  $('sourceExample').classList.toggle('selected',source==='example');
  $('sourceUpload').classList.toggle('selected',source==='upload');
  $('sourceExample').setAttribute('aria-pressed',String(source==='example'));
  $('sourceUpload').setAttribute('aria-pressed',String(source==='upload'));
  $('exampleOptions').hidden=source!=='example';
  $('uploadOptions').hidden=source!=='upload';
  $('oodWarning').hidden=source!=='upload';
  state.data=null;state.window=0;state.lastSignature='';
  clearPrediction();clearPreview();showError('');
  if(source==='example')analyze(false);
  else if(state.file)analyze(false);
}
function getSignature(){
  const f=state.file;
  return [state.source,f?.name,f?.size,f?.lastModified,
    document.querySelector('input[name="normalization"]:checked')?.value,$('stride').value].join('|');
}
function requestData(doPredict){
  const fd=new FormData();
  fd.append('source',state.source);
  fd.append('is_raw',String(state.source==='upload' && document.querySelector('input[name="normalization"]:checked')?.value==='raw'));
  fd.append('stride',String($('stride').value||100));
  fd.append('do_predict',String(doPredict));
  if(state.source==='upload'){
    if(!state.file)throw new Error('Hãy chọn file CSV, NPZ hoặc ZIP trước.');
    if(state.file.size>20*1024*1024)throw new Error('Tệp vượt quá giới hạn 20 MB.');
    fd.append('file',state.file,state.file.name);
  }
  return fd;
}
async function analyze(doPredict){
  if(state.busy)return;
  let form;
  try{form=requestData(doPredict);}catch(e){showError(e.message);return;}
  const signature=getSignature();showError('');setBusy(true,doPredict);
  try{
    const response=await fetch('/api/analyze',{method:'POST',body:form});
    let result;
    try{result=await response.json();}catch{throw new Error(`API trả về HTTP ${response.status}.`);}
    if(!response.ok)throw new Error(typeof result.detail==='string'?result.detail:'Có lỗi xảy ra khi phân tích.');
    const resetWindow=state.lastSignature!==signature;
    state.lastSignature=signature;state.data=result;
    if(resetWindow || state.window>=result.windows.length)state.window=0;
    updateAll();
  }catch(e){
    showError(e.message||'Không thể xử lý dữ liệu.');
    if(doPredict)setResultStatus('Không thể chạy mô hình',e.message||'Kiểm tra API/Render Logs.','error');
  }finally{setBusy(false);}
}
function clearPreview(){
  ['pointCount','windowCount','statStd','statPeak','statRms'].forEach(id=>txt(id,'—'));
  txt('signalComment','Chọn mẫu hoặc tải file để xem tín hiệu và thống kê.');
  const sel=$('windowSelect');sel.replaceChildren(new Option('Chưa có dữ liệu','0'));
  $('detailBody').textContent='Chưa có dữ liệu.';
  drawCharts();
}
function clearPrediction(){
  txt('finalPrediction','Chưa dự đoán');
  txt('finalDescription','Nhấn Run TF-C Prediction để chạy mô hình trên dữ liệu đã chọn.');
  txt('mainConfidence','—');
  $('confidenceFill').style.width='0%';
  txt('trueLabel','—');
  $('probabilityBars').replaceChildren(Object.assign(document.createElement('div'),{className:'no-results',textContent:'Điểm cho từng cử chỉ sẽ xuất hiện sau khi chạy TF-C.'}));
  $('matchBanner').hidden=true;
  $('downloadCsv').disabled=true;
  setResultStatus('Sẵn sàng phân tích','Hãy chọn dữ liệu rồi chạy dự đoán.');
}
function updateAll(){
  const info=state.data.info;
  txt('pointCount',Number(info.raw_rows).toLocaleString('vi-VN'));
  txt('windowCount',info.window_count);
  const notes=[...(info.notices||[]),`Đơn vị hiển thị: ${info.signal_units}`];
  if(info.sampling_hz)notes.push(`Tần số lấy mẫu ước lượng: ${fmt(info.sampling_hz)} Hz`);
  const list=document.createElement('ul');
  notes.forEach(line=>{const li=document.createElement('li');li.textContent=line;list.append(li);});
  $('detailBody').replaceChildren(list);
  const select=$('windowSelect');select.replaceChildren();
  state.data.windows.forEach((w,i)=>select.add(new Option(`Cửa sổ ${i+1} / ${state.data.windows.length}`,String(i))));
  const samples=$('sampleSelect');samples.replaceChildren();
  if(state.source==='example')state.data.windows.forEach((w,i)=>{
    samples.add(new Option(`${w.truth?.label||'Cử chỉ'} · SGR #${w.name}`,String(i)));
  });
  updateWindow();
  $('downloadCsv').disabled=!state.data.windows.some(w=>w.prediction);
}
function updateWindow(){
  if(!state.data?.windows?.length){clearPreview();clearPrediction();return;}
  const win=state.data.windows[state.window];
  $('windowSelect').value=String(state.window);
  if(state.source==='example')$('sampleSelect').value=String(state.window);
  txt('statStd',fmt(win.statistics.std));
  txt('statPeak',fmt(win.statistics.peak));
  txt('statRms',fmt(win.statistics.rms));
  txt('signalComment',win.statistics.comment);
  txt('trueLabel',win.truth?.label||'Không có nhãn');
  requestAnimationFrame(drawCharts);
  if(!win.prediction){
    clearPrediction();
    txt('trueLabel',win.truth?.label||'Không có nhãn');
    return;
  }
  const p=win.prediction;
  txt('finalPrediction',p.label);
  txt('finalDescription',`Cửa sổ ${state.window+1}/${state.data.windows.length} · Cử chỉ có điểm Softmax cao nhất.`);
  txt('mainConfidence',fmt(p.score*100,1)+'%');
  $('confidenceFill').style.width=`${Math.max(0,Math.min(100,p.score*100))}%`;
  const bars=$('probabilityBars');bars.replaceChildren();
  state.data.labels.forEach((label,i)=>{
    const row=document.createElement('div');row.className='prob-item'+(label.id===p.id?' is-top':'');
    const name=document.createElement('span');name.className='prob-name';name.textContent=label.label;
    const track=document.createElement('div');track.className='bar-track';
    const fill=document.createElement('div');fill.className='bar-fill';fill.style.width=`${Math.max(0,Math.min(100,p.probabilities[i]*100))}%`;
    track.append(fill);
    const percent=document.createElement('strong');percent.textContent=fmt(p.probabilities[i]*100,1)+'%';
    row.append(name,track,percent);bars.append(row);
  });
  const match=$('matchBanner');
  if(win.truth){
    const ok=p.matches_truth;
    match.hidden=false;match.classList.toggle('bad',ok===false);
    match.textContent=ok?'✓ Dự đoán khớp với nhãn tham chiếu.':'! Dự đoán khác nhãn tham chiếu.';
  }else match.hidden=true;
  setResultStatus('TF-C inference completed','Đã chạy mô hình trên dữ liệu thực.','success');
}
function setupCanvas(id){
  const canvas=$(id),rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;
  if(!w||!h)return null;
  const dpr=Math.min(2,window.devicePixelRatio||1);
  canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);return {ctx,w,h};
}
function emptyPlot(ctx,w,h){ctx.textAlign='center';ctx.fillStyle='#9ba6b7';ctx.font='12px system-ui, sans-serif';ctx.fillText('Chưa có dữ liệu',w/2,h/2);}
function drawFrame(ctx,w,h,domain,labelFormatter){
  const L=38,T=10,R=10,B=26,pw=w-L-R,ph=h-T-B;
  ctx.font='10px system-ui, sans-serif';ctx.lineWidth=1;
  for(let i=0;i<=3;i++){
    const y=T+i*ph/3;ctx.strokeStyle='#edf0f6';ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(L+pw,y);ctx.stroke();
    ctx.fillStyle='#a0aabd';ctx.textAlign='right';const span=Math.abs(domain.max-domain.min);const digits=span<.1?3:(span<1?2:1);ctx.fillText(fmt(domain.max+(domain.min-domain.max)*i/3,digits),L-7,y+3);
  }
  for(let i=0;i<=4;i++){
    const x=L+i*pw/4;ctx.fillStyle='#a1aabc';ctx.textAlign='center';ctx.fillText(labelFormatter(i/4),x,h-8);
  }
  return {L,T,R,B,pw,ph};
}
function drawTimeChart(){
  const canvas=setupCanvas('signalChart');if(!canvas)return;
  const {ctx,w,h}=canvas;const win=state.data?.windows?.[state.window];
  if(!win){emptyPlot(ctx,w,h);return;}
  const values=win.signal,n=values.length;
  if(n<2){emptyPlot(ctx,w,h);return;}
  let min=Infinity,max=-Infinity;
  for(const row of values)for(const v of row){min=Math.min(min,v);max=Math.max(max,v);}
  const pad=Math.max((max-min)*.12,.08);min-=pad;max+=pad;
  const times=win.time||[],hasHz=Boolean(state.data?.info?.sampling_hz);
  const scaleTimes=hasHz&&times.length===n&&times[n-1]>times[0];
  const range=scaleTimes?times[n-1]-times[0]:n-1;
  const f=drawFrame(ctx,w,h,{min,max},p=>scaleTimes?fmt(p*range,1):String(Math.round(p*(n-1))));
  for(let axis=0;axis<3;axis++){
    ctx.beginPath();ctx.strokeStyle=COLORS[axis];ctx.lineWidth=1.6;ctx.lineCap='round';ctx.lineJoin='round';
    values.forEach((row,i)=>{
      const ratio=scaleTimes?(times[i]-times[0])/range:i/(n-1);
      const x=f.L+ratio*f.pw,y=f.T+(max-row[axis])/(max-min)*f.ph;
      if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
    });
    ctx.stroke();
  }
}
function spectrum(signal){
  const n=signal.length;
  const magnitude=signal.map(a=>Math.hypot(a[0],a[1],a[2]));
  const mean=magnitude.reduce((a,b)=>a+b,0)/n;
  const windowed=magnitude.map((v,i)=>(v-mean)*(.5-.5*Math.cos(2*Math.PI*i/(n-1))));
  const power=[];
  for(let k=0;k<=Math.floor(n/2);k++){
    let re=0,im=0;
    for(let i=0;i<n;i++){
      const rad=2*Math.PI*k*i/n;re+=windowed[i]*Math.cos(rad);im-=windowed[i]*Math.sin(rad);
    }
    power.push(2*Math.hypot(re,im)/n);
  }
  return power;
}
function drawSpectrumChart(){
  const canvas=setupCanvas('spectrumChart');if(!canvas)return;
  const {ctx,w,h}=canvas,win=state.data?.windows?.[state.window];
  if(!win){emptyPlot(ctx,w,h);txt('spectrumUnits','Độ lớn FFT của gia tốc |a|');return;}
  const values=spectrum(win.signal),n=values.length;
  const highest=Math.max(...values,1e-7)*1.11;
  const hz=state.data?.info?.sampling_hz;
  const maxFreq=hz?hz/2:.5;
  txt('spectrumUnits',hz?'Tần số (Hz) · phổ mô tả':'Tần số (chu kỳ/mẫu) · phổ mô tả');
  const f=drawFrame(ctx,w,h,{min:0,max:highest},p=>fmt(p*maxFreq,hz?1:2));
  const barW=f.pw/n;
  ctx.fillStyle='#8687e8';
  for(let i=0;i<n;i++){
    const bh=Math.min(f.ph,values[i]/highest*f.ph);
    ctx.fillRect(f.L+i*barW,f.T+f.ph-bh,Math.max(.6,barW*.7),bh);
  }
}
function drawCharts(){drawTimeChart();drawSpectrumChart();}
function exportCsv(){
  if(!state.data?.windows?.some(w=>w.prediction))return;
  const heads=['window','prediction','prediction_vi','softmax_score','true_label',...state.data.labels.map(l=>'prob_'+l.name)];
  const rows=state.data.windows.filter(w=>w.prediction).map(w=>[
    w.index+1,w.prediction.name,w.prediction.label,w.prediction.score,w.truth?.name||'',...w.prediction.probabilities
  ]);
  const csv='\ufeff'+[heads,...rows].map(line=>line.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download='gesturelab_tfc_results.csv';document.body.append(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1500);
}
async function checkHealth(){
  try{
    const response=await fetch('/api/health');if(!response.ok)throw new Error('HTTP '+response.status);
    const health=await response.json();state.modelFound=Boolean(health.model_file_present);
    $('modelDot').className='status-dot '+(state.modelFound?'online':'offline');
    txt('modelStatus',state.modelFound?'TF-C model ready':'Chưa có checkpoint');
    if(!state.modelFound)showError('Không tìm thấy models/tfc_har_to_sgr_safe.pt. Có thể xem dữ liệu mẫu, nhưng chưa thể dự đoán thật.');
  }catch(e){$('modelDot').className='status-dot offline';txt('modelStatus','API chưa kết nối');showError('Không kết nối được tới FastAPI. Hãy kiểm tra server và URL.');}
}
function chooseFile(file){
  if(!file)return;
  state.file=file;state.data=null;state.window=0;state.lastSignature='';
  clearPreview();clearPrediction();
  $('fileName').hidden=false;txt('fileName',`${file.name} · ${(file.size/1024).toFixed(1)} KB`);
  analyze(false);
}
function listen(){
  $('sourceExample').addEventListener('click',()=>chooseSource('example'));
  $('sourceUpload').addEventListener('click',()=>chooseSource('upload'));
  $('sampleSelect').addEventListener('change',e=>{state.window=Number(e.target.value)||0;updateWindow();});
  $('windowSelect').addEventListener('change',e=>{state.window=Number(e.target.value)||0;updateWindow();});
  $('predictButton').addEventListener('click',()=>analyze(true));
  $('downloadCsv').addEventListener('click',exportCsv);
  $('uploadFile').addEventListener('change',e=>chooseFile(e.target.files?.[0]));
  const drop=$('dropZone');
  ['dragenter','dragover'].forEach(evt=>drop.addEventListener(evt,e=>{e.preventDefault();drop.classList.add('drag-over');}));
  ['dragleave','drop'].forEach(evt=>drop.addEventListener(evt,e=>{e.preventDefault();drop.classList.remove('drag-over');}));
  drop.addEventListener('drop',e=>chooseFile(e.dataTransfer?.files?.[0]));
  document.querySelectorAll('input[name="normalization"]').forEach(node=>node.addEventListener('change',()=>{if(state.source==='upload'&&state.file)analyze(false);}));
  $('stride').addEventListener('change',()=>{if(state.source==='upload'&&state.file)analyze(false);});
  let resizeTimer;
  window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(drawCharts,100);});
}
async function init(){listen();await checkHealth();await analyze(false);}
init();
