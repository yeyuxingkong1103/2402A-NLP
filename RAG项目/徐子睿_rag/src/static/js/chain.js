/* ============================================================
   static/js/chain.js —— 「检索」页签（检索链路透视）

   在链路中的位置：
       浏览器 → 本文件 → POST /api/search（只检索、不生成答案）

   这是 V2 迭代的核心产物：把黑盒的检索过程摊开展示，让人能回答"为什么是它命中"。
   渲染四块内容（数据全部来自后端返回的 trace 与 candidates，前端只展示、不计算）：
       1. 六个阶段的状态条（查询改写 → 语义召回 → 关键词召回 → RRF 融合 → 候选精排 → 上下文组织）
       2. 查询改写前后的对照
       3. 候选片段表（两路各自的分数与融合分，空分数表示该路未命中）
       4. 最终 Top-k 结果，标出每条是"LLM 精排"还是"RRF 回退"
   ============================================================ */

/* ================= 检索链路透视 ================= */
// 渲染一个带颜色的分数小标签（向量分 / BM25 分 / RRF 分 / 精排分）。
// val 为 null 时返回空串而不是显示 0 —— 后端用 null 表示"这一路没有召回这条候选"，
// 与"分数是 0"是完全不同的含义，界面上必须能区分
function scoreCell(label, cls, val) {
  return val == null ? '' : `<span class="score-cell ${cls}">${label} ${escapeHtml(val)}</span> `;
}
// 把 /api/search 的返回渲染成"检索链路透视"面板。这是 V2 迭代的核心产物：
// 把黑盒的检索过程摊开展示，让人能回答"为什么是它命中"。
// 分四块渲染：
//   1. 六个阶段的状态条（查询改写→语义召回→关键词召回→RRF 融合→候选精排→上下文组织），
//      从 trace 里按步骤名匹配，标出 完成/跳过/未执行 与耗时
//   2. 查询改写前后的对照（原文 → 补全术语后的检索串）
//   3. 候选片段表（两路各自的分数与融合分，空分数表示该路未命中）
//   4. 最终 Top-k 结果，并标出每条是"LLM 精排"还是"RRF 回退"
// 这些数据都来自后端返回的 trace 与 candidates，前端只做展示、不做计算
function renderChain(j) {
  const trace = j.trace || [];
  const traceSteps = ['查询改写','语义召回','关键词召回','RRF 融合','候选精排','上下文组织'];
  let html = '';
  for (const s of traceSteps) {
    const t = trace.find(x => x.step === s);
    html += `<div class="chain-stage">
      <div class="stage-head">
        <div class="idx ${t ? (t.status === 'done' ? 'green' : 'amber') : 'gray'}">${t ? (t.status === 'done' ? '✓' : '⊘') : '·'}</div>
        <div class="sname">${s}</div>
        <div class="sdet">${t ? `<b>${escapeHtml(t.status === 'done' ? '完成' : t.status === 'skipped' ? '跳过' : t.status)}</b> · ${escapeHtml(t.detail || '')}` : '未执行'}</div>
      </div>
    </div>`;
  }
  html += '<div class="card"><h2>查询改写</h2>';
  const rw = j.rewritten_query || '';
  html += `<div class="rewrite-flow"><span class="orig">${escapeHtml(j.query || '')}</span> <span class="arrow">→</span> <span class="terms">${escapeHtml(rw)}</span></div></div>`;

  const cands = j.candidates || [];
  html += '<div class="card"><h2>候选片段（融合打分）</h2>';
  if (cands.length) {
    html += `<table><thead><tr><th>#</th><th>页码</th><th>章节</th><th>语义类型</th><th>向量分</th><th>BM25 分</th><th>RRF 分</th><th>片段</th></tr></thead><tbody>` +
      cands.map(c => `<tr>
        <td><b>${escapeHtml(c.rrf_rank)}</b></td>
        <td>p${escapeHtml(c.page)}</td><td>${escapeHtml(c.section || '-')}</td>
        <td>${c.semantic_type ? `<span class="badge gray">${escapeHtml(c.semantic_type)}</span>` : '-'}</td>
        <td>${c.vec_score != null ? `<span class="score-cell vec">${escapeHtml(c.vec_score)}</span>` : '<span class="muted">—</span>'}</td>
        <td>${c.bm25_score != null ? `<span class="score-cell bm25">${escapeHtml(c.bm25_score)}</span>` : '<span class="muted">—</span>'}</td>
        <td><span class="score-cell rrf">${escapeHtml(c.rrf_score)}</span></td>
        <td class="muted" style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(c.text.slice(0, 55))}…</td>
      </tr>`).join('') + '</tbody></table>';
    html += `<div class="hint">空分数 = 该路未命中此候选（RRF 不要求双路都命中，加权求和取 top）</div>`;
  } else { html += '<div class="empty">无融合候选</div>'; }
  html += '</div>';

  const res = j.results || [];
  html += '<div class="card"><h2>检索结果（Top-k）</h2>';
  if (res.length) {
    html += res.map((r, i) => `<div class="mini-hit">
      <div class="row1">
        <span class="badge blue">#${i+1}</span>
        ${scoreCell('向量', 'vec', r.vec_score)}${scoreCell('BM25', 'bm25', r.bm25_score)}${scoreCell('RRF', 'rrf', r.rrf_score)}${scoreCell('精排', 'rr', r.rerank_score)}
        <span class="badge gray">p${escapeHtml(r.page)}</span>
        ${r.rerank_source ? `<span class="badge ${r.rerank_source === 'ollama' ? 'green' : 'amber'}">${r.rerank_source === 'ollama' ? 'LLM 精排' : 'RRF 回退'}</span>` : ''}
      </div>
      <div class="row2">${escapeHtml(r.section || '')}${r.semantic_type ? ' · ' + escapeHtml(r.semantic_type) : ''}</div>
      <div class="row2">${escapeHtml(r.text)}</div>
    </div>`).join('');
  } else { html += '<div class="empty">无检索结果（知识库为空或无有效检索词）</div>'; }
  html += '</div>';

  html += '<div class="hint" style="margin-bottom:16px">链路摘要：' + escapeHtml(j.note || '') + '</div>';
  $('chainResult').innerHTML = html;
}
// 执行一次检索（POST /api/search，非流式）。
// 这个接口只检索、不生成答案 —— 用途就是让人纯粹观察检索这一步的结果。
// k 由页面上的输入框决定；用 finally 恢复按钮状态，保证请求出错时按钮不会卡在禁用态
async function doSearch() {
  const q = $('searchQ').value.trim();
  if (!q) return;
  $('chainResult').innerHTML = '<div class="loading">跑全链路中…</div>';
  $('searchBtn').disabled = true;
  try {
    const j = await fetchJson('/api/search', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ q, k: parseInt($('searchK').value, 10) })
    });
    renderChain(j);
  } catch (err) { showEmpty($('chainResult'), err.message); }
  finally { $('searchBtn').disabled = false; }
}
$('searchBtn').onclick = doSearch;
$('searchQ').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

