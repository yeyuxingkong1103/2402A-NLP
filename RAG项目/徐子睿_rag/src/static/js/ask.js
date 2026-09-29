/* ============================================================
   static/js/ask.js —— 「问答」页签（流式问答）

   在链路中的位置：
       浏览器 → 本文件 → POST /api/ask/stream（SSE，事件序列 start → trace… → answer → done）

   三部分：
       DEMO_QUESTIONS   预置示例问题（演示/答辩时点一下即可跑）
       renderTrace      渲染右侧的流水线追踪面板
       doAsk            用 fetch + ReadableStream 手动解析 SSE 并逐步渲染答案与引用

   为什么不用浏览器内置的 EventSource：
       EventSource 只支持 GET、无法带请求体，而这个接口是 POST + JSON body，
       所以只能用 fetch 拿流、自己按 SSE 协议切帧。
   ============================================================ */

/* ================= 示例问答 ================= */
// 预置的示例问题（覆盖知识库里那几份电力国标文档的典型问法）。
// 放成常量的目的：演示/答辩时不必现场想问题，点一下就能跑；
// 也顺带充当一份"这个知识库能回答什么"的说明
const DEMO_QUESTIONS = [
  'SF6气体回收装置应具备哪些功能？',
  'SF6气体净化处理装置需要哪些功能？',
  '六氟化硫气体现场循环再利用的流程是什么？',
  'GB/T 44653-2024 适用于哪些设备和场景？',
  '变压器油中溶解气体的气相色谱测定方法是什么？',
  '绝缘油取样前需要做哪些准备工作？',
  '运行中变压器油的质量指标有哪些？',
  '特高压变压器分接开关的技术要求是什么？',
  '石油天然气输送管用宽厚钢板有哪些检验规则？',
  'SF6 气体的回收率要求是多少？',
];
// 把示例问题渲染成一排可点的标签，点一下即填入输入框并直接提问。
// 用立即执行函数（IIFE）包起来：给它一个独立作用域，避免里面用到临时变量时污染全局。
// data-q 存的是下标而不是问题全文 —— 问题里有引号和中文，塞进 HTML 属性容易出转义问题，
// 存下标就完全避开了这类麻烦
(function initDemoChips() {
  $('demoChips').innerHTML = DEMO_QUESTIONS.map((q, i) =>
    `<button class="q-chip" data-q="${i}"><span class="qnum">Q${i+1}</span>${escapeHtml(q)}</button>`).join('');
  $('demoChips').querySelectorAll('.q-chip').forEach(chip => {
    chip.onclick = () => { $('askQ').value = DEMO_QUESTIONS[Number(chip.dataset.q)]; doAsk(); };
  });
})();

// 渲染流水线追踪面板：每一步的名称、状态标记和耗时（毫秒）。
// 与 renderChain 的区别：那个是"检索页"的完整透视，这个是"问答页"侧边的精简追踪条。
// 状态标记用符号区分（✓ 完成 / ⊘ 跳过 / 其他状态原文），跳过的步骤也有记录 ——
// 这正是"链路可观测"的价值：能看出某一步是没执行，而不是默默失败
function renderTrace(trace) {
  $('tracePanel').innerHTML = (trace || []).map(t => {
    const ok = t.status === 'done' ? 'ok' : t.status === 'skipped' ? 'skip' : '';
    const mark = t.status === 'done' ? '✓' : t.status === 'skipped' ? '⊘' : escapeHtml(t.status);
    return `<div class="tstep">
      <div><span class="${ok}">${mark} ${escapeHtml(t.step)}</span>
      ${t.detail ? `<div class="det">${escapeHtml(t.detail)}</div>` : ''}</div>
      <span class="ms">${t.ms != null ? t.ms + ' ms' : ''}</span></div>`;
  }).join('') || '<div class="empty">无追踪数据</div>';
}
// 执行一次流式问答（POST /api/ask/stream，SSE）。
// 这里用 fetch + ReadableStream 手动解析 SSE，而不是用浏览器内置的 EventSource：
//   EventSource 只支持 GET、无法带请求体，而这个接口是 POST + JSON body，
//   所以只能用 fetch 拿流、自己按 SSE 协议切帧。
//
// SSE 解析的三个关键点：
//   1. 按 "\\n\\n" 切帧，buffer 里留下的最后一段可能是半帧，必须留到下一轮再拼
//      （网络分片不保证一个事件的字节一次到齐）
//   2. TextDecoder 的 { stream: true } —— 让解码器自己处理跨分片的多字节字符，
//      否则一个中文字被切在两个分片之间时会解成乱码
//   3. 每帧里找以 "data: " 开头的那行，切掉前缀（6 个字符）再 JSON.parse
//
// 事件处理（顺序即后端发送顺序）：
//   trace  逐条追加并即时重绘追踪面板 —— 这是"链路逐步点亮"效果的实现处
//   answer 拿到最终答案，渲染正文与引用；并按关键词判断是否为拒答，给不同样式
//   error  抛异常，走统一错误展示
//   done   提前跳出循环，不必等连接关闭
// 用 finalPayload 记录是否收到过 answer：收完却仍是 null 说明流程异常，
// 显式报错比显示一个空白答案好
async function doAsk() {
  const q = $('askQ').value.trim();
  if (!q) return;
  $('askResult').innerHTML = '<div class="loading">思考中…</div>';
  $('tracePanel').innerHTML = '<div class="loading">流水线运行中…</div>';
  $('askBtn').disabled = true;
  const trace = [];
  try {
    const response = await fetch(API + '/api/ask/stream', {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ q })
    });
    if (!response.ok || !response.body) {
      const text = await response.text();
      throw new Error(text || `请求失败：HTTP ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '', finalPayload = null;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';
      for (const part of parts) {
        const line = part.split('\n').find(row => row.startsWith('data: '));
        if (!line) continue;
        const event = JSON.parse(line.slice(6));
        if (event.type === 'trace' && event.data) { trace.push(event.data); renderTrace(trace); }
        else if (event.type === 'answer' && event.data) {
          finalPayload = event.data;
          const answer = finalPayload.answer || '';
          const refused = /未找到相关依据|知识库为空|LLM 调用失败/.test(answer);
          $('askResult').innerHTML =
            `<div class="answer ${refused ? 'refuse' : ''}">${escapeHtml(answer)}</div>` +
            (finalPayload.citations && finalPayload.citations.length ? `<div class="cite"><span class="cite-label">引用</span>${citeChips(finalPayload.citations)}</div>` : '');
        }
        else if (event.type === 'error' && event.data) throw new Error(event.data.message || '流式请求失败');
        else if (event.type === 'done') break;
      }
    }
    if (!trace.length) renderTrace([]);
    if (!finalPayload) throw new Error('未收到答案');
  } catch (err) {
    $('askResult').innerHTML = `<div class="answer refuse">${escapeHtml(err.message)}</div>`;
    renderTrace(trace);
  } finally { $('askBtn').disabled = false; }
}
$('askBtn').onclick = doAsk;
$('askQ').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doAsk(); } });
document.addEventListener('keydown', e => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); $('askQ').focus(); } });

