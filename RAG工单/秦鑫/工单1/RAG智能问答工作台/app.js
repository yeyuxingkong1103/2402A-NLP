const sampleAnswers={
  '武汉兴图新科参与制定了哪个技术标准？':'根据《招股说明书 1.pdf》，公司参与制定了国家标准《信息安全技术 网络安全等级保护基本要求》，并承担了相关技术研究工作。',
  '公司注册资本是多少？':'公司注册资本为 7,370.00 万元。该数据来自招股说明书中“基本情况”章节。',
  '电子信息行业的下游主要包括哪些行业？':'电子信息行业的下游主要包括国防军工、政府、能源、交通、金融及其他需要信息化解决方案的行业。',
  '武汉兴图新科来自军用领域的收入分别是多少？':'报告期内，公司来自军用领域的收入如下：2019 年 1-6 月 4,627.14 万元；2018 年度 18,780.67 万元；2017 年度 14,414.16 万元；2016 年度 6,464.51 万元。'
};
const navItems=[...document.querySelectorAll('.nav-item')];
const views=[...document.querySelectorAll('.view')];
const titleMap={workspace:'问答工作台',knowledge:'知识库',evaluation:'检索质量评估',docs:'文档与演示'};
function switchView(view){views.forEach(v=>v.classList.toggle('active-view',v.id===`${view}-view`));navItems.forEach(n=>n.classList.toggle('active',n.dataset.view===view));document.getElementById('page-title').textContent=titleMap[view]||'问答工作台';window.scrollTo({top:0,behavior:'smooth'})}
navItems.forEach(n=>n.addEventListener('click',()=>switchView(n.dataset.view)));
document.querySelectorAll('[data-view]').forEach(n=>{if(!n.classList.contains('nav-item'))n.addEventListener('click',()=>switchView(n.dataset.view))});
const toast=document.getElementById('toast');
function showToast(message){toast.textContent=message;toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),2600)}
const messages=document.getElementById('chat-messages');const input=document.getElementById('question-input');
function addMessage(type,text){const wrap=document.createElement('div');wrap.className=`message ${type}`;const avatar=type==='assistant'?'<div class="message-avatar">R</div>':'<div class="message-avatar user-avatar">你</div>';wrap.innerHTML=type==='assistant'?`${avatar}<div class="bubble answer-bubble"><div class="bubble-meta">RAG 助手 <span>刚刚 · 1.2s</span></div><p>${text}</p><div class="sources"><span>↗ 来源  招股说明书 1.pdf · 相关片段</span><span>相关度 93%</span></div><div class="answer-actions"><button>□ 复制</button><button>♡ 有帮助</button></div></div>`:`<div class="bubble"><div class="bubble-meta">你 <span>刚刚</span></div><p>${text}</p></div>${avatar}`;messages.appendChild(wrap);messages.scrollTop=messages.scrollHeight}
document.getElementById('question-form').addEventListener('submit',e=>{e.preventDefault();const q=input.value.trim();if(!q)return;addMessage('user',q);input.value='';setTimeout(()=>{const key=Object.keys(sampleAnswers).find(k=>q.includes(k)||k.includes(q));addMessage('assistant',key?sampleAnswers[key]:'我已在文档中完成检索，但没有找到足够直接的证据。请尝试补充时间范围、公司名称或业务关键词。');},450)});
document.querySelectorAll('.suggestions button').forEach(b=>b.addEventListener('click',()=>{input.value=b.dataset.question;input.focus()}));
document.getElementById('reset-btn').addEventListener('click',()=>{showToast('会话已重置，知识库保持不变');messages.scrollTop=0});
const fileInput=document.getElementById('file-input');
function chooseFile(){fileInput.click()}
document.getElementById('upload-btn').addEventListener('click',chooseFile);document.getElementById('upload-btn-2').addEventListener('click',chooseFile);document.querySelector('.upload-zone').addEventListener('click',e=>{if(!e.target.closest('button'))chooseFile()});
fileInput.addEventListener('change',()=>{const file=fileInput.files[0];if(file){showToast(`已接收 ${file.name}，开始解析并建立索引`);document.getElementById('chat-status').textContent=`正在解析「${file.name}」`;setTimeout(()=>{document.getElementById('chat-status').textContent=`已连接到「${file.name}」知识库`;showToast('解析完成，文档已加入知识库')},1200)}});
document.getElementById('demo-btn').addEventListener('click',()=>{switchView('workspace');showToast('演示已开始：示例问题已准备就绪')});
document.querySelectorAll('.answer-actions button').forEach(b=>b.addEventListener('click',()=>showToast(b.textContent.includes('复制')?'回答已复制到剪贴板':'感谢反馈')));
