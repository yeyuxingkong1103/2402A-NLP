const navItems=[...document.querySelectorAll('.nav-item')];
const views=[...document.querySelectorAll('.view')];
const titleMap={workspace:'问答工作台',knowledge:'知识库',evaluation:'检索质量评估',docs:'文档与演示'};
const messages=document.getElementById('chat-messages');
const input=document.getElementById('question-input');
const toast=document.getElementById('toast');
let lastQuestion='';
let sessionId=`工单5-${crypto.randomUUID?.()||Date.now()}`;

function showToast(message){toast.textContent=message;toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),2800)}
function switchView(view){views.forEach(item=>item.classList.toggle('active-view',item.id===`${view}-view`));navItems.forEach(item=>item.classList.toggle('active',item.dataset.view===view));document.getElementById('page-title').textContent=titleMap[view]||titleMap.workspace;window.scrollTo({top:0,behavior:'smooth'});if(view==='evaluation')loadEvaluation()}
navItems.forEach(item=>item.addEventListener('click',()=>switchView(item.dataset.view)));
document.querySelectorAll('[data-view]').forEach(item=>{if(!item.classList.contains('nav-item'))item.addEventListener('click',()=>switchView(item.dataset.view))});

function textNode(tag,text,className){const el=document.createElement(tag);if(className)el.className=className;el.textContent=text;return el}
function addMessage(type,text,sources=[],metrics={},tableSources=[],imageSources=[]){
  const wrap=document.createElement('div');wrap.className=`message ${type}`;
  if(type==='assistant'){
    const avatar=textNode('div','R','message-avatar');
    const bubble=document.createElement('div');bubble.className='bubble answer-bubble';
    bubble.append(textNode('div',`RAG 助手 · ${metrics.elapsed_ms??'-'} ms`,'bubble-meta'));
    text.split('\n').filter(Boolean).forEach(line=>bubble.append(textNode('p',line.replace(/^-\s*/,''))));
    if(sources.length){const sourceBar=document.createElement('div');sourceBar.className='sources';sourceBar.append(textNode('span',`来源片段 ${sources.length} 条`),textNode('span',`候选 ${metrics.candidate_count??'-'} 条`));bubble.append(sourceBar);sources.slice(0,3).forEach(source=>{const citation=textNode('div',`↗ ${source.document} · 第 ${source.page} 页 · ${source.snippet}`,'citation');bubble.append(citation)})}
    if(tableSources.length){const tableBar=document.createElement('div');tableBar.className='sources';tableBar.append(textNode('span',`表格证据 ${tableSources.length} 张`),textNode('span','pdfplumber 表格解析'));bubble.append(tableBar);tableSources.slice(0,2).forEach(source=>{const citation=textNode('div',`▦ ${source.document} · 第 ${source.page} 页 · 表格 ${source.table} · ${source.rows.slice(0,2).map(row=>row.join(' | ')).join('；')}`,'citation');bubble.append(citation)})}
    if(imageSources.length){const imageBar=document.createElement('div');imageBar.className='sources';imageBar.append(textNode('span',`图像证据 ${imageSources.length} 张`),textNode('span','PDF 图像解析'));bubble.append(imageBar);imageSources.slice(0,2).forEach(source=>{const citation=textNode('div',`▧ ${source.document} · 第 ${source.page} 页 · ${source.name} · ${source.width||'?'}×${source.height||'?'} · ${source.bytes} bytes`,'citation');bubble.append(citation)})}
    const actions=document.createElement('div');actions.className='answer-actions';const copy=textNode('button','□ 复制');copy.addEventListener('click',()=>navigator.clipboard?.writeText(text).then(()=>showToast('回答已复制')).catch(()=>showToast('复制失败，请手动选择文本')));actions.append(copy,textNode('button',sources.length?'♡ 有帮助':'检查问题后重试'));bubble.append(actions);wrap.append(avatar,bubble);
    if(metrics.elapsed_ms!==undefined){document.getElementById('retrieval-candidates').textContent=metrics.candidate_count??'-';document.getElementById('retrieval-latency').textContent=`${metrics.elapsed_ms}ms`;document.getElementById('latency-stat').textContent=`${metrics.elapsed_ms}ms`;const parsed=metrics.query_understanding||{};document.getElementById('intent-value').textContent=parsed.intent||'文档事实检索';const terms=(parsed.entities||[]).slice(0,4).map(entity=>textNode('span',entity,'chip'));terms.push(textNode('span',parsed.strategy||'BM25 + RRF','chip'));document.getElementById('query-entities').replaceChildren(...terms)}
  }else{
    const bubble=document.createElement('div');bubble.className='bubble';bubble.append(textNode('div','你 · 刚刚','bubble-meta'),textNode('p',text));wrap.append(bubble,textNode('div','你','message-avatar user-avatar'));
  }
  messages.append(wrap);messages.scrollTop=messages.scrollHeight;
}
async function ask(question){
  const q=question.trim();if(!q)return;lastQuestion=q;addMessage('user',q);input.value='';input.disabled=true;
  try{const response=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,top_k:5,session_id:sessionId})});const data=await response.json();if(!response.ok)throw new Error(data.error||'检索请求失败');sessionId=data.session_id||sessionId;addMessage('assistant',data.answer,data.sources||[],data.metrics||{},data.table_sources||[],data.image_sources||[])}
  catch(error){addMessage('assistant',`无法连接本地检索服务：${error.message}\n请确认已通过 run.bat 启动系统。`)}
  finally{input.disabled=false;input.focus()}
}
document.getElementById('question-form').addEventListener('submit',event=>{event.preventDefault();ask(input.value)});
document.querySelectorAll('.suggestions button').forEach(button=>button.addEventListener('click',()=>{input.value=button.dataset.question;input.focus()}));
document.getElementById('reset-btn').addEventListener('click',()=>{messages.replaceChildren();sessionId=`工单5-${crypto.randomUUID?.()||Date.now()}`;addMessage('assistant','会话已重置。输入问题后，我会基于当前 PDF 知识库检索并显示页码引用。');showToast('会话已重置')});

const fileInput=document.getElementById('file-input');
function chooseFile(){fileInput.click()}
document.getElementById('upload-btn').addEventListener('click',chooseFile);
document.getElementById('upload-btn-2').addEventListener('click',chooseFile);
document.querySelector('.upload-zone').addEventListener('click',event=>{if(!event.target.closest('button'))chooseFile()});
fileInput.addEventListener('change',async()=>{
  const file=fileInput.files[0];if(!file)return;
  document.getElementById('chat-status').textContent=`正在解析「${file.name}」`;
  document.getElementById('index-state').textContent='解析中';showToast('正在解析 PDF 并建立索引…');
  try{const form=new FormData();form.append('file',file);const response=await fetch('/api/upload',{method:'POST',body:form});const data=await response.json();if(!response.ok)throw new Error(data.error||'上传失败');document.getElementById('chat-status').textContent=`已连接到「${file.name}」知识库`;document.getElementById('kb-document').textContent=file.name;document.getElementById('kb-document-state').textContent='已解析并建立检索索引';document.getElementById('chunk-count').textContent=data.total_chunks.toLocaleString();document.getElementById('index-state').textContent='已完成';showToast(`解析完成，新增 ${data.chunks_added} 个片段`)}
  catch(error){document.getElementById('index-state').textContent='解析失败';document.getElementById('chat-status').textContent='PDF 解析失败';showToast(error.message)}finally{fileInput.value=''}
});

async function refreshStatus(){
  try{const response=await fetch('/api/status');if(!response.ok)throw new Error('索引服务异常');const data=await response.json();document.getElementById('kb-status').textContent=data.ready?'本地索引已就绪':'请上传 PDF 建立知识库';document.getElementById('chunk-count').textContent=data.chunks.toLocaleString();document.getElementById('index-state').textContent=data.ready?'已完成':'待上传';document.getElementById('chat-status').textContent=data.ready?`已连接到 ${data.documents.length} 个 PDF 文档`:'知识库为空';document.getElementById('index-progress').style.width=data.ready?'100%':'0%';document.getElementById('document-summary').textContent=`${data.documents.length} 个文档 · ${data.chunks.toLocaleString()} 个文本片段`;const rows=document.getElementById('document-rows');rows.replaceChildren();rows.append(textNode('div','文档名称　　文本片段　　状态','table-row table-header'));data.documents.forEach(name=>{const row=document.createElement('div');row.className='table-row';row.append(textNode('span',`PDF  ${name}`),textNode('span','-'),textNode('span',data.chunks.toLocaleString()),textNode('span','已索引','table-status'));rows.append(row)});if(data.documents[0])document.getElementById('kb-document').textContent=data.documents[0]}
  catch{document.getElementById('kb-status').textContent='服务未连接';document.getElementById('index-state').textContent='未启动';document.getElementById('chat-status').textContent='请先运行 run.bat 启动本地服务'}
}

async function loadEvaluation(){
  const box=document.getElementById('evaluation-rows');box.replaceChildren(textNode('div','正在运行 10 道示例题…','loading-row'));
  try{const response=await fetch('/api/evaluate');const data=await response.json();if(!response.ok)throw new Error(data.error||'评测失败');const score=document.getElementById('eval-after');score.textContent=String(Math.round(data.after_rate*10));score.append(textNode('span','/10'));document.getElementById('eval-before').textContent=`${Math.round(data.before_rate*10)}/10`;document.getElementById('eval-hit').textContent=`${Math.round(data.after_rate*10)}/10`;document.getElementById('eval-latency').textContent=`${data.after_avg_ms}ms`;document.getElementById('eval-progress').style.width=`${data.after_rate*100}%`;document.getElementById('eval-caption').textContent=data.metric;box.replaceChildren();data.cases.forEach(item=>{const row=document.createElement('div');row.className='evaluation-row';row.append(textNode('span',String(item.id),'case-id'),textNode('span',item.question,'case-question'),textNode('span',item.before_hit?'命中':'未命中',item.before_hit?'case-hit':'case-miss'),textNode('span',item.after_hit?'命中':'未命中',item.after_hit?'case-hit':'case-miss'));row.append(textNode('small',item.matched_page?`命中相关页 ${item.matched_page} · ${item.matched_snippet}`:'没有命中相关页','case-evidence'));box.append(row)})}
  catch(error){box.replaceChildren(textNode('div',`评测加载失败：${error.message}`,'loading-row'))}
}
document.getElementById('refresh-evaluation').addEventListener('click',loadEvaluation);
document.getElementById('open-evaluation').addEventListener('click',event=>{event.preventDefault();switchView('evaluation')});
document.getElementById('demo-btn').addEventListener('click',()=>{switchView('workspace');input.value='报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？';input.focus();showToast('演示问题已填入输入框')});
document.querySelectorAll('.answer-actions button').forEach(button=>button.addEventListener('click',()=>showToast('感谢反馈')));
refreshStatus();
