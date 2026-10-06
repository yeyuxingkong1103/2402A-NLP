/* 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
   工单01 - 基于PDF文档的问答系统 */

const $ = (s) => document.querySelector(s);
const api = (p) => p;

let lastAsk = null;      // {question, answer, citations} —— 供反馈使用
let es = null;           // 当前 SSE 连接
let mediaRecorder = null;
let audioChunks = [];

/* ---------------- 标签页 ---------------- */
document.querySelectorAll('.tab').forEach(t => {
  t.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.page').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    $('#page-' + t.dataset.page).classList.add('active');
    if (t.dataset.page === 'kb') { loadKb(); pollIngest(); }
    if (t.dataset.page === 'eval') { loadReport(); }
  };
});

/* ---------------- 健康检查 ---------------- */
async function health() {
  const dot = $('#dot');
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    dot.className = 'status-dot ' + (d.ok ? 'ok' : 'err');
    const warn = $('#boot-warning');
    if (!d.ok) {
      warn.innerHTML = `<div class="alert err">
        <b>部分依赖不可用</b><br>
        Ollama：${d.ollama}<br>Milvus：${d.milvus}<br>
        <span class="hint">Milvus 未启动时无法检索。启动命令：
        <code>docker compose -f docker-compose.milvus.yml up -d</code></span>
      </div>`;
    } else if (d.collection_rows === 0) {
      warn.innerHTML = `<div class="alert warn">
        <b>知识库为空</b> —— 请到「知识库」页执行一次入库。
      </div>`;
    } else {
      warn.innerHTML = '';
    }
  } catch (e) {
    dot.className = 'status-dot err';
    $('#boot-warning').innerHTML =
      `<div class="alert err">无法连接后端：${e}</div>`;
  }
}
health();
setInterval(health, 20000);

/* ---------------- 提问（SSE 流式） ---------------- */
function setMeta(el, d) {
  const ttft = d.ttft_ms || 0;
  const cls = ttft && ttft <= 3000 ? 'ok' : 'warn';
  el.innerHTML =
    `<b class="${cls}">TTFT ${ttft} ms</b>（验收口径 ≤3000ms） ｜ ` +
    `完整答案 <b>${d.total_ms || 0} ms</b>` +
    (d.retrieval_ms != null ? ` ｜ 检索 ${d.retrieval_ms} ms` : '') +
    (d.n_candidates ? ` ｜ 候选 ${d.n_candidates}，冗余过滤 ${d.n_filtered}` : '');
}

function renderCitations(cites, target) {
  target.innerHTML = '';
  (cites || []).forEach(c => {
    const s = document.createElement('span');
    s.className = 'cite';
    s.textContent = `${c.page_label}${c.chunk_type === 'table' ? '（表）' : ''}`;
    s.title = (c.section_path ? c.section_path + '\n' : '') + (c.snippet || '');
    s.onclick = () => showEvidence(cites);
    target.appendChild(s);
  });
}

function showEvidence(cites) {
  $('#evidence-card').style.display = 'block';
  $('#evidence').innerHTML = (cites || []).map((c, i) => `
    <div class="snippet">
      <span class="lbl">[片段${i + 1}] 页码 ${c.page_label} ｜ 相似度 ${c.score}
        ｜ 类型 ${c.chunk_type}${c.section_path ? ' ｜ ' + c.section_path : ''}</span>
      ${escapeHtml(c.snippet || '')}
    </div>`).join('');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, m =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[m]));
}

function ask() {
  const question = $('#q').value.trim();
  if (!question) return;
  if (es) { es.close(); es = null; }

  $('#answer-card').style.display = 'block';
  $('#compare-card').style.display = 'none';
  $('#answer').textContent = '';
  $('#answer').classList.add('streaming');
  $('#citations').innerHTML = '';
  $('#feedback').style.display = 'none';
  $('#fb-msg').textContent = '';
  $('#meta').textContent = '检索中…';
  $('#mode-tag').textContent = 'RAG';
  $('#ask').disabled = true;

  const k = $('#topk').value;
  let cites = [];
  es = new EventSource(api(`/api/chat/stream?question=${encodeURIComponent(question)}&top_k=${k}`));

  es.addEventListener('status', e => {
    $('#meta').textContent = JSON.parse(e.data).message;
  });

  es.addEventListener('meta', e => {
    const d = JSON.parse(e.data);
    if (d.mode === 'no_rag') { $('#mode-tag').textContent = '纯 LLM'; }
    cites = d.citations || [];
    renderCitations(cites, $('#citations'));
  });

  es.addEventListener('token', e => {
    $('#meta').textContent = '生成中…';
    $('#answer').textContent += JSON.parse(e.data).text;
  });

  es.addEventListener('done', e => {
    const d = JSON.parse(e.data);
    $('#answer').classList.remove('streaming');
    setMeta($('#meta'), d);
    $('#feedback').style.display = 'flex';
    lastAsk = {
      question,
      answer: $('#answer').textContent,
      citations: cites
    };
    es.close(); es = null;
    $('#ask').disabled = false;
  });

  es.addEventListener('error', e => {
    try {
      const d = JSON.parse(e.data);
      $('#meta').innerHTML = `<span class="warn">${escapeHtml(d.message)}</span>`;
    } catch (_) {
      $('#meta').innerHTML = `<span class="warn">连接中断</span>`;
    }
    $('#answer').classList.remove('streaming');
    es && es.close(); es = null;
    $('#ask').disabled = false;
  });

  es.onerror = () => {
    if (es && es.readyState === EventSource.CLOSED) {
      $('#answer').classList.remove('streaming');
      $('#ask').disabled = false;
    }
  };
}

/* ---------------- 对比纯 LLM ---------------- */
async function compare() {
  const question = $('#q').value.trim();
  if (!question) return;
  $('#compare-card').style.display = 'block';
  $('#norag-answer').textContent = '请求中…';
  $('#rag-answer').textContent = '请求中…';
  $('#norag-meta').textContent = '';
  $('#rag-meta').textContent = '';

  try {
    const [a, b] = await Promise.all([
      fetch('/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, no_rag: true })
      }).then(r => r.json()),
      fetch('/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, no_rag: false, top_k: Number($('#topk').value) })
      }).then(r => r.json())
    ]);
    $('#norag-answer').textContent = a.answer || a.detail || '(空)';
    $('#rag-answer').textContent = b.answer || b.detail || '(空)';
    setMeta($('#norag-meta'), a);
    setMeta($('#rag-meta'), b);
    renderCitations(b.citations, $('#citations'));
    $('#answer-card').style.display = 'block';
    $('#answer').textContent = b.answer || '';
    $('#mode-tag').textContent = 'RAG';
  } catch (e) {
    $('#rag-answer').textContent = '请求失败：' + e;
  }
}

/* ---------------- 反馈 ---------------- */
$('#feedback').addEventListener('click', async (ev) => {
  const btn = ev.target.closest('button[data-r]');
  if (!btn || !lastAsk) return;
  try {
    const r = await fetch('/api/feedback', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: lastAsk.question, answer: lastAsk.answer,
        rating: btn.dataset.r, citations: lastAsk.citations
      })
    });
    const d = await r.json();
    $('#fb-msg').textContent = `已记录（累计 ${d.received} 条）`;
    document.querySelectorAll('#feedback button').forEach(b => b.classList.remove('on'));
    btn.classList.add('on');
  } catch (e) { $('#fb-msg').textContent = '记录失败'; }
});

/* ---------------- 语音输入 ---------------- */
async function toggleRec() {
  const btn = $('#rec');
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    return;
  }
  const avail = await fetch('/api/asr/available').then(r => r.json()).catch(() => ({ available: false }));
  if (!avail.available) {
    alert('语音功能未安装（可选依赖）：\n\n' + (avail.hint || ''));
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    alert('无法访问麦克风：' + e.message);
    return;
  }
  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => e.data.size && audioChunks.push(e.data);
  mediaRecorder.onstop = async () => {
    btn.classList.remove('recording');
    btn.textContent = '🎤 录音';
    stream.getTracks().forEach(t => t.stop());
    const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
    btn.disabled = true;
    btn.textContent = '识别中…';
    const fd = new FormData();
    fd.append('file', blob, 'rec.webm');
    try {
      const d = await fetch('/api/asr', { method: 'POST', body: fd }).then(r => r.json());
      if (d.text) { $('#q').value = d.text; }
      else { alert('未识别到语音内容'); }
    } catch (e) { alert('识别失败：' + e); }
    btn.disabled = false;
    btn.textContent = '🎤 录音';
  };
  mediaRecorder.start();
  btn.classList.add('recording');
  btn.textContent = '⏹ 停止';
}

/* ---------------- 评估页 ---------------- */
async function runEval() {
  $('#run-eval').disabled = true;
  $('#eval-msg').textContent = '启动中…';
  const q = `with_judge=${$('#opt-judge').checked}&with_no_rag=${$('#opt-norag').checked}`;
  try {
    await fetch(`/api/evaluate/run?${q}`, { method: 'POST' });
  } catch (e) { $('#eval-msg').textContent = '启动失败：' + e; return; }
  pollEval();
}

async function pollEval() {
  const d = await fetch('/api/evaluate/status').then(r => r.json()).catch(() => null);
  if (!d) return;
  const pct = d.total ? Math.round(d.done / d.total * 100) : 0;
  $('#eval-progress > i').style.width = pct + '%';
  $('#eval-msg').textContent = `${d.message}${d.error ? ' ｜ ' + d.error : ''}`;
  if (d.running) { setTimeout(pollEval, 2000); $('#run-eval').disabled = true; }
  else { $('#run-eval').disabled = false; if (d.done) loadReport(); }
}

async function loadReport() {
  try {
    const r = await fetch('/api/evaluate/report');
    if (!r.ok) { $('#eval-msg').textContent = '尚无评估报告'; return; }
    renderReport(await r.json());
  } catch (e) { /* 静默 */ }
}

function pct(v) { return v == null ? '—' : (v * 100).toFixed(0) + '%'; }

function renderReport(rep) {
  const s = rep.summary;
  $('#eval-summary-card').style.display = 'block';
  $('#eval-table-card').style.display = 'block';
  $('#eval-detail-card').style.display = 'block';

  const kpi = (v, k, cls = '') =>
    `<div class="kpi"><div class="v ${cls}">${v}</div><div class="k">${k}</div></div>`;

  $('#eval-kpis').innerHTML = [
    kpi(`${s.rag_rule_hits}/${s.n}`, 'RAG 规则化数值命中', 'ok'),
    kpi(`${s.no_rag_rule_hits}/${s.n}`, '纯 LLM 规则命中', 'err'),
    kpi(pct(s.avg_context_relevance), '上下文相关性'),
    kpi(pct(s.avg_faithfulness), '忠实度（无幻觉）'),
    kpi(pct(s.avg_answer_relevance), '答案相关性'),
    kpi(pct(s.avg_context_recall), '上下文召回'),
    kpi(pct(s.avg_answer_correctness), '答案正确性'),
    kpi(s.ttft_p50_ms + ' ms', 'TTFT P50'),
    kpi(s.ttft_p95_ms + ' ms', 'TTFT P95'),
    kpi(s.total_p50_ms + ' ms', '完整答案 P50'),
    kpi(s.total_p95_ms + ' ms', '完整答案 P95'),
  ].join('');

  const tb = $('#eval-table tbody');
  tb.innerHTML = rep.items.map(it => `
    <tr>
      <td class="num">${it.id}</td>
      <td>${escapeHtml(it.question)}</td>
      <td>${it.rule_hit ? '<span class="badge ok">命中</span>' : '<span class="badge err">未命中</span>'}</td>
      <td>${it.no_rag_rule_hit ? '<span class="badge ok">命中</span>' : '<span class="badge err">未命中</span>'}</td>
      <td class="num">${pct(it.context_relevance)}</td>
      <td class="num">${pct(it.faithfulness)}</td>
      <td class="num">${pct(it.answer_relevance)}</td>
      <td class="num">${pct(it.context_recall)}</td>
      <td class="num">${pct(it.answer_correctness)}</td>
      <td class="num">${it.ttft_ms}</td>
      <td class="num">${it.total_ms}</td>
      <td>${(it.citations || []).join(' ')}</td>
    </tr>`).join('');

  $('#eval-detail').innerHTML = rep.items.map(it => `
    <details>
      <summary>[${it.id}] ${escapeHtml(it.question)}
        ${it.rule_hit ? '✅' : '❌'} ${escapeHtml(it.rule_detail || '')}</summary>
      <div class="grid2">
        <div>
          <p class="panel-title">RAG 回答</p>
          <div class="answer">${escapeHtml(it.rag_answer)}</div>
          <div class="meta">引用：${(it.citations || []).join(' ') || '—'}</div>
        </div>
        <div>
          <p class="panel-title">纯 LLM 回答</p>
          <div class="answer">${escapeHtml(it.no_rag_answer || '（未跑）')}</div>
        </div>
      </div>
      <p class="panel-title" style="margin-top:12px">人工参考答案（${escapeHtml(it.evidence_page || '')}）</p>
      <div class="answer">${escapeHtml(it.reference_answer || '')}</div>
    </details>`).join('');
}

/* ---------------- 知识库页 ---------------- */
async function loadKb() {
  try {
    const d = await fetch('/api/kb').then(r => r.json());
    $('#kb-table tbody').innerHTML = (d.collections || []).map(c =>
      `<tr><td>${escapeHtml(c.name)}</td><td class="num">${c.rows}</td></tr>`).join('')
      || '<tr><td colspan="2" class="hint">（无）</td></tr>';
  } catch (e) {
    $('#kb-table tbody').innerHTML = `<tr><td colspan="2" class="hint">读取失败：${e}</td></tr>`;
  }
}

async function doIngest() {
  $('#do-ingest').disabled = true;
  $('#ingest-msg').textContent = '启动中…';
  try {
    const r = await fetch('/api/ingest', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rebuild: $('#opt-rebuild').checked })
    });
    if (!r.ok) { $('#ingest-msg').textContent = '启动失败：' + (await r.text()); }
  } catch (e) { $('#ingest-msg').textContent = '启动失败：' + e; }
  pollIngest();
}

async function pollIngest() {
  const d = await fetch('/api/ingest/status').then(r => r.json()).catch(() => null);
  if (!d) return;
  const pctv = d.total ? Math.round(d.done / d.total * 100) : (d.stage === 'parse' ? 5 : 0);
  $('#ingest-progress > i').style.width = pctv + '%';
  $('#ingest-msg').textContent =
    `[${d.stage}] ${d.message}${d.error ? ' ｜ ' + d.error : ''}`;
  if (d.running) { $('#do-ingest').disabled = true; setTimeout(pollIngest, 1500); }
  else {
    $('#do-ingest').disabled = false;
    if (d.stage === 'done') { loadKb(); }
  }
}

async function loadPage() {
  const n = $('#page-no').value.trim();
  if (!n) return;
  $('#page-out').textContent = '读取中…';
  try {
    const d = await fetch(`/api/kb/page/${n}`).then(r => r.json());
    if (!d.items || !d.items.length) {
      $('#page-out').innerHTML = '<div class="alert warn">该页无入库片段</div>';
      return;
    }
    $('#page-out').innerHTML = d.items.map(it => `
      <div class="snippet">
        <span class="lbl">#${it.chunk_index} ｜ ${it.page_label} ｜ ${it.chunk_type}
          ${it.section_path ? ' ｜ ' + escapeHtml(it.section_path) : ''}</span>
        ${escapeHtml(it.content)}
      </div>`).join('');
  } catch (e) { $('#page-out').textContent = '读取失败：' + e; }
}

/* ---------------- 绑定 ---------------- */
$('#ask').onclick = ask;
$('#compare').onclick = compare;
$('#rec').onclick = toggleRec;
$('#run-eval').onclick = runEval;
$('#load-report').onclick = loadReport;
$('#do-ingest').onclick = doIngest;
$('#reload-kb').onclick = loadKb;
$('#load-page').onclick = loadPage;

$('#en').onchange = (e) => {
  $('#q').value = e.target.checked
    ? 'What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?'
    : '报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？';
};

$('#q').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); ask(); }
});

$('#q').value = '报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？';
