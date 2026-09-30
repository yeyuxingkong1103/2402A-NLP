/* ============================================================
   static/js/kb.js —— 「知识库」页签

   在链路中的位置：
       浏览器 → 本文件 → /api/kb/overview、/api/documents、/api/kb/chunks、/api/kb/document

   包含两部分：
       1. 顶部统计卡（文档数 / chunk 数 / 总页数 / 向量数）
       2. 文档表：逐份展开 chunk 明细（手风琴，同时只展开一份）、删除文档

   注意统计数据的来源：
       chunk 数与文档数来自 /api/kb/overview（扫描 Milvus 的真实存量），
       总页数来自 /api/documents（构建时写下的登记表）—— 两者口径不同，不能混用。
   ============================================================ */

/* ================= 知识库统计 ================= */
// 渲染顶部四张统计卡：文档数、chunk 数、总页数、向量数。
// 两个接口并行拉取（Promise.all）—— 它们互不依赖，串行会白白多等一个往返。
// 页数需要自己累加：/api/kb/overview 只给每份文档的页数，没有总数
async function loadStats() {
  try {
    const [ov, docs] = await Promise.all([
      fetchJson('/api/kb/overview'),
      fetchJson('/api/documents'),
    ]);
    const totalPages = (docs.documents || []).reduce((s, d) => s + (Number(d.pages) || 0), 0);
    $('kbStats').innerHTML = [
      ['📄', '文档数', ov.total_docs, 'accent-blue'],
      ['🧩', 'chunk 数', ov.total_points, 'accent-green'],
      ['📃', '总页数', totalPages, 'accent-amber'],
      ['🧬', '向量数', ov.total_points, 'accent-violet'],
    ].map(([ic, l, n, c]) => `<div class="stat-card ${c}"><div class="num"><span class="ic">${ic}</span>${escapeHtml(n)}</div><div class="lbl">${escapeHtml(l)}</div></div>`).join('');
  } catch (err) {
    $('kbStats').innerHTML = `<div class="card" style="grid-column:1/-1"><div class="empty">${escapeHtml(err.message)}</div></div>`;
  }
}

/* ================= 知识库 ================= */
// 渲染知识库页签：先刷新统计卡，再列出文档（每行可展开 chunk 明细、可删除）。
// 数据源是 /api/kb/overview，它以 Milvus 的实际内容为准 ——
// 所以这里看到的 chunk 数与"检索时真正能命中多少内容"是一致的
async function loadKb() {
  loadStats();
  const tb = $('kbBody');
  try {
    const j = await fetchJson('/api/kb/overview');
    const docs = j.documents || [];
    if (!docs.length) { tb.innerHTML = '<tr><td colspan="6" class="empty">知识库为空（请先在「构建链路」上传 PDF）</td></tr>'; return; }
    tb.innerHTML = docs.map((d, i) => `<tr class="kb-doc-row" data-source="${escapeHtml(d.filename)}">
      <td><span class="kb-caret">▶</span>${escapeHtml(d.filename)}</td><td>${escapeHtml(d.chunks)}</td><td>${escapeHtml(d.pages ?? '-')}</td>
      <td>${fmtSize(d.size)}</td><td>${escapeHtml(d.time ?? '-')}</td>
      <td><button class="op-btn view" data-action="view" data-index="${i}">展开</button>
          <button class="op-btn del" data-action="delete" data-index="${i}">删除</button></td></tr>`).join('');
    tb.querySelectorAll('button[data-action]').forEach(btn => {
      const doc = docs[Number(btn.dataset.index)];
      btn.onclick = e => {
        e.stopPropagation();
        if (btn.dataset.action === 'view') toggleDoc(doc.filename);
        else delDoc(doc.filename);
      };
    });
    tb.querySelectorAll('tr.kb-doc-row').forEach((tr, i) => {
      tr.onclick = () => toggleDoc(docs[i].filename);
    });
  } catch (err) { tb.innerHTML = `<tr><td colspan="6" class="empty">${escapeHtml(err.message)}</td></tr>`; }
}
// 当前展开的是哪份文档（手风琴效果：同时只展开一份）。null 表示全收起
let _kbOpenSource = null;
// chunk 明细缓存：{文件名: chunk 数组}。展开过一次后再收起/展开就不必重新请求。
// 注意删除文档时要同步清掉对应缓存（见 delDoc），否则会显示已删文档的旧内容
const _kbChunkCache = {};
// 展开/收起某份文档的 chunk 明细行。
// 展开时在文档行下方动态插入一个详情行（colspan=6 铺满整行），
// 里面按 chunk 依次列出编号、页码、章节、语义类型、关键词和正文
async function toggleDoc(source) {
  const tb = $('kbBody');
  const rows = Array.from(tb.querySelectorAll('tr.kb-doc-row'));
  const findTr = s => rows.find(r => r.dataset.source === s);
  const tr = findTr(source);
  const isOpen = _kbOpenSource === source;
  // 收起旧的（手风琴：同时只展开一个）
  if (!isOpen && _kbOpenSource) {
    const oldTr = findTr(_kbOpenSource);
    if (oldTr) {
      oldTr.classList.remove('expanded');
      const ob = oldTr.querySelector('button[data-action="view"]');
      if (ob) ob.textContent = '展开';
      const det = oldTr.nextElementSibling;
      if (det && det.classList.contains('kb-detail-row')) det.remove();
    }
    _kbOpenSource = null;
  }
  if (isOpen) { // 再点收起
    if (tr) {
      tr.classList.remove('expanded');
      const b = tr.querySelector('button[data-action="view"]');
      if (b) b.textContent = '展开';
      const det = tr.nextElementSibling;
      if (det && det.classList.contains('kb-detail-row')) det.remove();
    }
    _kbOpenSource = null;
    return;
  }
  if (!tr) return;
  tr.classList.add('expanded');
  const b = tr.querySelector('button[data-action="view"]');
  if (b) b.textContent = '收起';
  _kbOpenSource = source;
  // 行下方插入展开行
  const det = document.createElement('tr');
  det.className = 'kb-detail-row';
  det.innerHTML = '<td colspan="6"><div class="kb-detail-wrap"><div class="kb-detail-load">正在加载 chunk 明细…</div></div></td>';
  tr.after(det);
  const wrap = det.querySelector('.kb-detail-wrap');
  try {
    let chunks = _kbChunkCache[source];
    if (!chunks) {
      const j = await fetchJson('/api/kb/chunks?source=' + encodeURIComponent(source));
      chunks = j.chunks || [];
      _kbChunkCache[source] = chunks;
    }
    if (_kbOpenSource !== source) return; // 已被用户收起/切换
    const head = `<div class="kb-detail-head"><b>${escapeHtml(source)}</b><span class="kb-count">共 ${chunks.length} 块 · 点击任一块展开全文</span></div>`;
    const body = chunks.length
      ? chunks.map(c => `<div class="chunk-box">
        <div class="meta">${escapeHtml(c.chunk_id)} · 第${escapeHtml(c.page)}页 · ${escapeHtml(c.section)}${c.semantic_type ? ' · <span class="badge gray">' + escapeHtml(c.semantic_type) + '</span>' : ''}${c.important_kwd && c.important_kwd.length ? ' · 关键词 ' + escapeHtml(c.important_kwd.join('/')) : ''}</div>
        <div class="kb-chunk-text">${escapeHtml(c.text)}</div>
        <button class="kb-chunk-more">展开全文 ▾</button></div>`).join('')
      : '<div class="empty">暂无 chunk</div>';
    wrap.innerHTML = head + body;
    bindChunkToggles(wrap);
  } catch (err) {
    if (_kbOpenSource !== source) return;
    wrap.innerHTML = `<div class="kb-detail-err">加载失败：${escapeHtml(err.message)}</div>`;
  }
}
// 给展开区里每个 chunk 卡片绑定"展开全文/收起全文"。
// 同时绑在卡片本身和那个按钮上，且按钮的 handler 要 stopPropagation ——
// 不阻止冒泡的话点按钮会触发两次（按钮一次、冒泡到卡片一次），
// 结果展开后立刻又收起，表现为"点了没反应"
function bindChunkToggles(wrap) {
  wrap.querySelectorAll('.chunk-box').forEach(box => {
    const toggle = () => {
      const open = box.classList.toggle('open');
      box.querySelector('.kb-chunk-more').textContent = open ? '收起全文 ▴' : '展开全文 ▾';
    };
    box.onclick = toggle;
    box.querySelector('.kb-chunk-more').onclick = e => {
      e.stopPropagation();
      toggle();
    };
  });
}
// 删除一份文档。确认框里明确写出"同时删除向量和存档 PDF"——
// 这是不可逆操作，必须让用户知道影响范围后再确认。
// 删除后清掉前端缓存与展开状态，再重新拉取列表
async function delDoc(source) {
  if (!confirm(`确定从知识库删除「${source}」？\n将同时删除其全部向量和存档 PDF。`)) return;
  try {
    const j = await fetchJson('/api/kb/document?source=' + encodeURIComponent(source), { method: 'DELETE' });
    if (!j.ok) throw new Error(j.error || '删除失败');
    if (_kbOpenSource === source) _kbOpenSource = null;
    delete _kbChunkCache[source];
    loadKb(); loadDocs();
  } catch (err) { alert(err.message); }
}

