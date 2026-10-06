/* ============================================================
   static/js/build.js —— 页签切换与「文档构建」页签

   在链路中的位置：
       浏览器 → 本文件 → backend/server 的 /api/upload、/api/build/status、/api/documents

   包含两部分：
       1. 全局页签切换（导航栏按钮 ↔ section#tab-* 的显示）
       2. 文档构建页签：拖拽/选择 PDF → 上传 → 轮询构建进度 → 展示已入库文档表
   ============================================================ */

/* ================= 页签切换 ================= */
// 给导航栏每个按钮绑定切换逻辑：先清掉所有按钮和 section 的 active，再给当前这对加上。
// data-tab 的值与 section 的 id 通过约定对应（data-tab="build" -> section#tab-build），
// 所以新增页签只要补一对 data-tab / id，不用改这里。
// 切换时按需拉取数据（构建页拉文档列表、知识库页拉统计），而不是在页面加载时全量预取 ——
// 减少首屏请求数，也让每次切回来都看到最新数据
document.querySelectorAll('nav button').forEach(b => {
  b.onclick = () => {
    document.querySelectorAll('nav button').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.page').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    $('tab-' + b.dataset.tab).classList.add('active');
    if (b.dataset.tab === 'build') loadDocs();
    if (b.dataset.tab === 'kb') loadKb();
  };
});

/* ================= 构建链路 ================= */
// chosenFile 保存"待上传的 PDF"。两种选文件的方式（点击选择、拖拽）都写进它，
// 上传按钮据此启用/禁用
let chosenFile = null;
const dz = $('dropzone'), fi = $('fileInput');
dz.onclick = () => fi.click();   // 点拖拽区 = 点隐藏的文件选择框
fi.onchange = () => { chosenFile = fi.files[0]; $('uploadBtn').disabled = !chosenFile; };
// 三件事都要 preventDefault：浏览器对拖拽有默认行为（直接打开被拖入的文件），
// 不阻止的话页面会被那个 PDF 顶掉
['dragover','dragleave','drop'].forEach(ev => dz.addEventListener(ev, e => e.preventDefault()));
dz.addEventListener('dragover', () => dz.classList.add('drag'));   // 进入时高亮
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));// 离开时取消高亮
dz.addEventListener('drop', e => {
  dz.classList.remove('drag');
  // 只接受 .pdf。前端这一层过滤是为了给即时反馈，后端仍会再校验一次（不可只依赖前端）
  chosenFile = Array.from(e.dataTransfer.files).find(f => f.name.toLowerCase().endsWith('.pdf')) || null;
  fi.value = '';   // 清空 input，避免"上次选的文件"和这次拖入的混在一起
  $('uploadBtn').disabled = !chosenFile;
  if (!chosenFile) alert('请拖入 PDF 文件');
});
// 轮询构建进度的定时器句柄。存成模块级变量是为了能在"上传新文件"和"构建结束"时清掉它，
// 否则会产生多个并行的轮询
let pollTimer = null;
// 上传按钮：提交文件后立刻转入"轮询进度"模式。
// 上传接口是立即返回的（后端在后台线程里跑构建），所以这里不能等响应就以为完事了，
// 必须靠轮询 /api/build/status 来获知进度
$('uploadBtn').onclick = async () => {
  if (!chosenFile) return;
  $('buildCard').style.display = 'block';
  $('steps').innerHTML = '<div class="loading">准备上传…</div>';
  $('buildLoading').style.display = 'block';
  $('uploadBtn').disabled = true;
  const fd = new FormData();
  fd.append('file', chosenFile);
  try {
    const j = await fetchJson('/api/upload', { method: 'POST', body: fd });
    if (!j.ok) throw new Error(j.error || '上传失败');
    clearInterval(pollTimer);
    pollTimer = setInterval(pollStatus, 800);
    pollStatus();
  } catch (err) { showEmpty($('steps'), err.message); $('buildLoading').style.display = 'none'; $('uploadBtn').disabled = false; }
};
// 查一次构建进度并刷新步骤列表（每 800ms 被 pollTimer 调一次）。
// 结束条件看 j.running：后端在构建完成或失败时把它置为 false，
// 此时清掉定时器、恢复按钮、并刷新文档列表与统计数字。
// 失败分支单独处理（catch）也要清定时器 —— 否则后端挂了会永远轮询下去
async function pollStatus() {
  try {
    const j = await fetchJson('/api/build/status');
    const box = $('steps');
    box.innerHTML = '';
    (j.steps || []).forEach(s => {
      const d = document.createElement('div');
      d.className = 'step ' + escapeHtml(s.status);
      d.innerHTML = `<div class="dot">${s.status === 'done' ? '✓' : s.status === 'error' ? '!' : ''}</div>
        <div class="name">${escapeHtml(s.step)}</div><div class="detail">${escapeHtml(s.detail)}</div><div class="time">${escapeHtml(s.t)}</div>`;
      box.appendChild(d);
    });
    if (!j.running) {
      clearInterval(pollTimer); pollTimer = null;
      $('buildLoading').style.display = 'none';
      $('uploadBtn').disabled = false;
      loadDocs(); loadStats();
    }
  } catch (err) {
    clearInterval(pollTimer); pollTimer = null;
    $('buildLoading').style.display = 'none';
    $('uploadBtn').disabled = false;
    showEmpty($('steps'), err.message);
  }
}
// 拉取已登记文档并渲染成表格（GET /api/documents）。
// 这一份数据来自 documents.json 登记表（页数/块数/构建耗时），
// 与"知识库"页签那份来自 Milvus 实测数据的总览是两回事 ——
// 登记表是构建时写下的快照，Milvus 那份是当前真实存量
async function loadDocs() {
  const tb = $('docBody');
  try {
    const j = await fetchJson('/api/documents');
    const docs = j.documents || [];
    if (!docs.length) { tb.innerHTML = '<tr><td colspan="7" class="empty">暂无文档（可拖入 PDF 构建）</td></tr>'; return; }
    tb.innerHTML = docs.map(d => `<tr>
      <td>${escapeHtml(d.filename)}</td><td>${fmtSize(d.size)}</td><td>${escapeHtml(d.pages)}</td>
      <td>${escapeHtml(d.chunks)}</td><td>${escapeHtml(d.vectors)}</td><td>${escapeHtml(d.seconds)}s</td><td>${escapeHtml(d.time)}</td></tr>`).join('');
  } catch (err) { tb.innerHTML = `<tr><td colspan="7" class="empty">${escapeHtml(err.message)}</td></tr>`; }
}
$('refreshBtn').onclick = () => { loadDocs(); };

