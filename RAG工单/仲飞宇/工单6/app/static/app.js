/* 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
   工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
   工单编号：人工智能NLP-RAG-Query 理解优化任务
   工单01 - 基于PDF文档的问答系统
   工单02 - 基于PDF文档的问答系统的优化
   工单05 - Query 理解优化（多轮对话）

   【工单05 的改动原则】**纯追加**：单轮问答页的 ask()/compare() 一行没动 ——
   它们承载工单01-04 的既有演示，重构的风险远大于收益。
   多轮走独立的一套：askTurn() / renderRewrite() / turnStore。 */

const $ = (s) => document.querySelector(s);
const api = (p) => p;

let lastAsk = null;      // {question, answer, citations} —— 供反馈使用
let es = null;           // 当前 SSE 连接
let metaInfo = null;     // 工单02：meta 事件里的检索侧信息（剖面/合并/邻块/上下文字数）
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
  // 工单02：把「这次用了哪套检索策略、它做了什么」直接摊在界面上 ——
  // 演示时不用口头解释，观众自己能看到"合并了几条重复表述、补了几条邻块"。
  let extra = '';
  if (d.profile) {
    const label = { baseline: '朴素基线', delivered: '工单01 交付版',
                    optimized: '工单02 优化版', fulltext: '全文检索',
                    hybrid: '混合检索', hybrid_weighted: '混合·加权',
                    hybrid_milvus: '混合·Milvus原生', optimized_hybrid: '混合+工单02',
                    hybrid_llm: '混合+LLM重排' }[d.profile] || d.profile;
    extra += ` ｜ 剖面 <b>${label}</b>`;
  }
  // 工单06：把这一轮实际走的检索路摊开
  if (d.retrieval_mode) {
    extra += ` ｜ 模式 <b>${d.retrieval_mode}</b>`;
    if (d.fusion) extra += `·${d.fusion}`;
    if (d.reranker) extra += ` ｜ 重排 <b>${d.reranker}</b>`;
    if (d.n_from_keyword) extra += ` ｜ 两路 向量${d.n_from_dense}/关键词${d.n_from_keyword}`;
    if (d.feedback_n) extra += ` ｜ 反馈${d.feedback_n}条`;
    if (d.rerank_fallback) extra += ` <b class="warn">（重排已回退）</b>`;
  }
  if (d.n_merged) extra += ` ｜ 同事实合并 ${d.n_merged}`;
  if (d.n_added_neighbors) extra += ` ｜ 邻块补充 ${d.n_added_neighbors}`;
  if (d.context_chars) extra += ` ｜ 上下文 ${d.context_chars} 字`;
  // 工单03：实体路由。两份招股书章节结构雷同，观众要能一眼看到"这题被限定到了哪份文档"。
  // 没过滤（没点名 / 一句话点了两家）时也要显示，否则会误以为路由坏了。
  if (d.routed_doc) {
    extra += ` ｜ 路由 <b>${d.routed_doc}</b>（命中「${d.route_matched}」）`;
    if (d.route_fallback) extra += ` <b class="warn">↩ 回退全库</b>`;
  } else if (d.profile) {
    extra += ` ｜ 路由 <b>不过滤</b>（未点名公司 / 跨文档）`;
  }
  // 【必须显式标注命中缓存】缓存里存的是**首次**那次的时间，可能是冷启动的 8 秒。
  // 不标注就等于把"首次冷启动"当成"当前耗时"展示，演示时会当场翻车。
  if (d.cached) {
    extra += ` ｜ <b class="warn">命中缓存（以上为首次耗时，本次几乎瞬时）</b>`;
  }
  el.innerHTML =
    `<b class="${cls}">TTFT ${ttft} ms</b>（验收口径 ≤3000ms） ｜ ` +
    `完整答案 <b>${d.total_ms || 0} ms</b>` +
    (d.retrieval_ms != null ? ` ｜ 检索 ${d.retrieval_ms} ms` : '') +
    (d.n_candidates ? ` ｜ 候选 ${d.n_candidates}，冗余过滤 ${d.n_filtered}` : '') +
    extra;
}

function renderCitations(cites, target) {
  target.innerHTML = '';
  (cites || []).forEach(c => {
    const s = document.createElement('span');
    s.className = 'cite';
    s.textContent = `${c.page_label}${{ table: '（表）', image: '（图）' }[c.chunk_type] || ''}`;
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
  const profile = $('#profile') ? $('#profile').value : 'delivered';
  let cites = [];
  es = new EventSource(api(
    `/api/chat/stream?question=${encodeURIComponent(question)}`
    + `&top_k=${k}&profile=${encodeURIComponent(profile)}` + modeQuery()));

  es.addEventListener('status', e => {
    $('#meta').textContent = JSON.parse(e.data).message;
  });

  es.addEventListener('meta', e => {
    const d = JSON.parse(e.data);
    if (d.mode === 'no_rag') { $('#mode-tag').textContent = '纯 LLM'; }
    else if (d.profile) { $('#mode-tag').textContent = 'RAG · ' + d.profile; }
    cites = d.citations || [];
    renderCitations(cites, $('#citations'));
    // 检索侧的元信息在 meta 事件里（done 事件只报时间），先存下来，
    // 等 done 到了一起渲染，避免中途把计时行覆盖掉。
    metaInfo = d;
  });

  es.addEventListener('token', e => {
    $('#meta').textContent = '生成中…';
    $('#answer').textContent += JSON.parse(e.data).text;
  });

  es.addEventListener('done', e => {
    const d = JSON.parse(e.data);
    $('#answer').classList.remove('streaming');
    // 把 meta 事件里的检索侧信息（剖面/合并/邻块/上下文字数）并进来一起显示
    setMeta($('#meta'), Object.assign({}, metaInfo || {}, d));
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

/* 工单06：把检索模式/权重/融合/重排器拼成 query（空值 = 跟随剖面） */
function modeQuery() {
  const q = [];
  const m = $('#rmode') && $('#rmode').value;
  if (m) q.push(`retrieval_mode=${encodeURIComponent(m)}`);
  const f = $('#rfusion') && $('#rfusion').value;
  if (m === 'hybrid' && f) {
    if (f === 'milvus') { q.push('fusion=rrf'); q.push('fusion_impl=milvus'); }
    else q.push(`fusion=${encodeURIComponent(f)}`);
  }
  const w = $('#rweight');
  if (m === 'hybrid' && w) q.push(`w_keyword=${w.value}`);
  const r = $('#rrerank') && $('#rrerank').value;
  if (r) q.push(`reranker=${encodeURIComponent(r)}`);
  return q.length ? '&' + q.join('&') : '';
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
        body: JSON.stringify({
          question, no_rag: false, top_k: Number($('#topk').value),
          profile: $('#profile') ? $('#profile').value : 'delivered'
        })
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
  // 工单02：评估也按剖面跑，评估页才能直接产出「优化前后」的对比数字
  const prof = $('#eval-profile') ? $('#eval-profile').value : 'delivered';
  const q = `with_judge=${$('#opt-judge').checked}`
          + `&with_no_rag=${$('#opt-norag').checked}`
          + `&profile=${encodeURIComponent(prof)}`;
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
    // ---- 工单02 检索侧指标：把"检索漏了"和"生成漏了"分开 ----
    kpi(s.profile || '—', '检索剖面'),
    kpi(pct(s.avg_context_keyword_coverage), 'CKC 上下文关键词覆盖'),
    kpi(pct(s.avg_page_precision), '检索页精确率'),
    kpi(pct(s.avg_page_recall), '检索页召回率'),
    kpi(s.avg_context_chars + ' 字', '平均上下文字数'),
    kpi(pct(s.avg_context_relevance), '上下文相关性'),
    kpi(pct(s.avg_faithfulness), '忠实度（无幻觉）'),
    kpi(pct(s.avg_answer_relevance), '答案相关性'),
    kpi(pct(s.avg_context_recall), '上下文召回'),
    kpi(pct(s.avg_answer_correctness), '答案正确性'),
    kpi(s.ttft_p50_ms + ' ms', 'TTFT P50'),
    // n=10 时 P95 就是最大值，单个离群点就能拉爆，所以并列给出达标题数
    kpi(`${s.ttft_under_3s ?? '—'}/${s.n}`, 'TTFT ≤3s 的题数', 'ok'),
    kpi(s.ttft_p95_ms + ' ms', 'TTFT P95（=最大值）'),
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
      <td>${it.routed_doc ? escapeHtml(it.routed_doc.replace('.pdf', ''))
            + (it.route_fallback ? ' <b class="warn">↩回退</b>' : '') : '—'}</td>
      <td class="num">${pct(it.context_keyword_coverage)}</td>
      <td class="num">${pct(it.page_precision)}</td>
      <td class="num">${pct(it.page_recall)}</td>
      <td class="num">${it.context_chars || '—'}</td>
      <td class="num">${it.n_merged || 0}/${it.n_added_neighbors || 0}</td>
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
          <div class="meta">路由：${it.routed_doc
            ? escapeHtml(it.routed_doc) + `（命中「${escapeHtml(it.route_matched || '')}」）`
              + (it.route_fallback ? ' <b class="warn">↩ 已回退全库</b>' : '')
            : '不过滤（未点名公司 / 跨文档）'}</div>
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

/* ======================================================================
   工单05 · 多轮对话（Query 理解）
   ====================================================================== */

/* 会话 id 放 sessionStorage（**不是** localStorage）：
   两个浏览器标签页天然拿到各自的 id —— 这正好是「会话隔离」演示的素材。 */
let sessionId = sessionStorage.getItem('sid') || null;
let turnIdx = 0;
const turnStore = new Map();   // turnIdx -> {question, answer, citations} 每轮独立存（反馈用）

const SCRIPT = [
  '报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？',
  '他参与的哪个工程荣获了国家科技进步一等奖？',
  '这个公司的法定代表人是谁？',
  '那武汉力源信息技术股份有限公司呢？',
  '武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？',
];

function initChatPage() {
  if ($('#script-buttons').childElementCount) return;
  $('#script-buttons').innerHTML = SCRIPT.map((q, i) =>
    `<button class="ghost script-btn" data-i="${i}">
       <b>Q${i + 1}</b> ${escapeHtml(q)}</button>`).join('');
  $('#script-buttons').addEventListener('click', ev => {
    const b = ev.target.closest('button[data-i]');
    if (b) { $('#cq').value = SCRIPT[+b.dataset.i]; $('#cq').focus(); }
  });
  if (sessionId) $('#sess-id').textContent = sessionId.slice(0, 8) + '…';
  else newSession();
}

function newSession() {
  sessionId = (window.crypto && crypto.randomUUID)
    ? crypto.randomUUID() : 's-' + Date.now() + '-' + Math.random().toString(16).slice(2);
  sessionStorage.setItem('sid', sessionId);
  turnIdx = 0; turnStore.clear();
  $('#chat-log').innerHTML = '';
  $('#sess-id').textContent = sessionId.slice(0, 8) + '…';
  $('#sess-info').textContent = `新会话已建立（${sessionId}）。同一标签页内连续追问会带上上下文；换标签页 = 换会话。`;
}

function bubble(role, text) {
  const el = document.createElement('div');
  el.className = 'bubble ' + role;
  el.innerHTML = role === 'bot'
    ? `<div class="rw meta"></div><div class="answer"></div>
       <div class="cites citations"></div><div class="meta bmeta"></div>
       <div class="feedback fb" style="display:none">
         <button data-r="up">👍 有帮助</button>
         <button data-r="down">👎 不准确</button>
       </div>`
    : `<div class="who">你</div><div class="answer">${escapeHtml(text)}</div>`;
  $('#chat-log').appendChild(el);
  $('#chat-log').scrollTop = $('#chat-log').scrollHeight;
  return el;
}

const METHOD_LABEL = {
  'none': '未改写（完整问题）',
  'rule-coref': '规则 · 代词消解',
  'rule-switch': '规则 · 话题切换',
  'rule-ellipsis': '规则 · 省略补全',
  'llm': 'LLM 兜底改写',
  'llm-failed': '⚠ 兜底失败（已按原问题检索）',
};

function renderRewrite(el, rw) {
  if (!el || !rw) return;
  const label = METHOD_LABEL[rw.method] || rw.method;
  const arrow = rw.rewritten && rw.rewritten !== rw.original
    ? `<b>${escapeHtml(rw.original)}</b> → <span class="rw-new">${escapeHtml(rw.rewritten)}</span>`
    : `<b>${escapeHtml(rw.original)}</b>（无需改写）`;
  el.innerHTML =
    `<span class="rewrite-chip ${rw.method}">${escapeHtml(label)}</span> ${arrow}` +
    (rw.carried_intent
      ? `<div class="hint">继承上一轮问点：<b>${escapeHtml(rw.carried_intent)}</b></div>` : '') +
    (rw.evidence ? `<div class="hint">${escapeHtml(rw.evidence)}</div>` : '');
}

function askTurn() {
  const q = $('#cq').value.trim();
  if (!q) return;
  if (!sessionId) newSession();
  const idx = turnIdx++;
  bubble('user', q);
  const bot = bubble('bot', '');
  bot.querySelector('.rw').textContent = '改写中…';
  $('#cq').value = '';
  $('#csend').disabled = true;

  const k = $('#topk').value;
  const profile = $('#profile') ? $('#profile').value : 'optimized';
  const src = `/api/chat/stream?question=${encodeURIComponent(q)}`
    + `&top_k=${k}&profile=${encodeURIComponent(profile)}`
    + `&session_id=${encodeURIComponent(sessionId)}`;
  const es = new EventSource(src);
  let cites = [];
  let meta = null;

  es.addEventListener('meta', e => {
    const d = JSON.parse(e.data);
    meta = d;
    cites = d.citations || [];
    renderRewrite(bot.querySelector('.rw'), d.rewrite);
    renderCitations(cites, bot.querySelector('.cites'));
    if (d.session_expired) {
      const warn = document.createElement('div');
      warn.className = 'alert warn';
      warn.innerHTML = '<b>会话已过期（TTL）</b> —— 上下文已失效，本轮按单轮处理并新建了会话。';
      bot.insertBefore(warn, bot.firstChild.nextSibling);
    }
  });

  es.addEventListener('token', e => {
    bot.querySelector('.answer').textContent += JSON.parse(e.data).text;
    $('#chat-log').scrollTop = $('#chat-log').scrollHeight;
  });

  es.addEventListener('done', e => {
    const d = Object.assign({}, meta || {}, JSON.parse(e.data));
    setMeta(bot.querySelector('.bmeta'), d);
    const fb = bot.querySelector('.fb');
    fb.style.display = 'flex';
    fb.dataset.turn = idx;
    turnStore.set(idx, {
      question: q,
      answer: bot.querySelector('.answer').textContent,
      citations: cites,
    });
    $('#csend').disabled = false;
    es.close();
  });

  es.addEventListener('error', e => {
    let msg = '连接中断';
    try { msg = JSON.parse(e.data).message; } catch (_) {}
    bot.querySelector('.rw').innerHTML = `<span class="warn">${escapeHtml(msg)}</span>`;
    $('#csend').disabled = false;
    es.close();
  });
}

/* 每轮各自的反馈：用事件委托 + turnStore，而不是再造一个 lastAsk */
$('#chat-log').addEventListener('click', async ev => {
  const btn = ev.target.closest('button[data-r]');
  if (!btn) return;
  const fb = btn.closest('.fb');
  const t = turnStore.get(Number(fb.dataset.turn));
  if (!t) return;
  try {
    await fetch('/api/feedback', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: t.question, answer: t.answer, rating: btn.dataset.r,
        citations: t.citations, session_id: sessionId,
        turn_index: Number(fb.dataset.turn) + 1,
      }),
    });
    fb.querySelectorAll('button').forEach(b => b.classList.remove('on'));
    btn.classList.add('on');
  } catch (e) { /* 静默：反馈不是主流程 */ }
});

async function showSessionStatus() {
  try {
    const d = await fetch(`/api/session/${encodeURIComponent(sessionId)}`).then(r => r.json());
    $('#sess-info').textContent = d.exists
      ? `会话 ${d.session_id.slice(0, 8)}… ｜ 轮次 ${d.n_turns} ｜ 焦点实体 ${d.focus_entity || '（无）'}`
        + ` ｜ 问点模板 ${d.intent_template || '（无）'} ｜ history ${d.history_chars} 字`
        + `（发送给模型的最近轮 ${d.history_turns} 轮）｜ 存库会话 ${d.store.n_sessions} 个`
      : `会话 ${sessionId.slice(0, 8)}… 不存在（可能已过期或被逐出）`;
  } catch (e) { $('#sess-info').textContent = '读取失败：' + e; }
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

/* 工单05 多轮对话页 */
$('#csend').onclick = askTurn;
$('#new-session').onclick = newSession;
$('#sess-status').onclick = showSessionStatus;
$('#cq-clear').onclick = () => {
  $('#chat-log').innerHTML = ''; turnStore.clear(); turnIdx = 0;
  $('#sess-info').textContent = '已清空界面（会话仍保留，下一轮仍带上下文）。';
};
$('#cq').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); askTurn(); }
});
initChatPage();

$('#en').onchange = (e) => {
  $('#q').value = e.target.checked
    ? 'What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?'
    : '报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？';
};

$('#rweight').addEventListener('input', e => {
  $('#rweight-v').textContent = e.target.value;
});

$('#q').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); ask(); }
});

$('#q').value = '报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？';
