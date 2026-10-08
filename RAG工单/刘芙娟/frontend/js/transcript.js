/* 结果渲染：答案正文区 + 引用片段区。
 *
 * 契约见 specs/006-medical-qa-input/contracts/sse.md 与 data-model.md §4。
 *
 * ⚠️ 两条渲染纪律，都不是风格问题：
 *
 * 1. renderCitations **按传入顺序渲染，绝不排序**。
 *    排序依据是余弦相似度，那是**服务端算出来的**（docs/05 §4.2）。前端重排
 *    等于把排序规则复刻了一遍，两份实现必然漂移 —— 而且漂移时没有任何测试
 *    能发现：服务端测试全绿，前端只是显示顺序不同。所以这里是 map，不是 sort。
 *
 * 2. appendToken 只追加，不重建 DOM。
 *    重建会让"先渲染后重排"成为可能，而那正是 FR-030 要防的。
 */

const ANSWER_JOINER = '\n\n';

/* 引用原文的预览长度。
 *
 * 这个数字不是随手定的：specs/005 实测过 chunk 的 text 最长约 4,094 字符
 * （7,794 字节，表格类 chunk 尤其长）。按 docs/05 §8 的 top_k = 3 算，
 * 三条引用全量展开约有 12,000 字符 —— 足够把引用卡片撑到十几屏，
 * 让"按相似度排序"这件事在视觉上彻底失效（用户根本翻不到第二条）。
 *
 * 所以默认只给预览，点「展开原文」才铺开。这也正好对上了 docs/01 §F3
 * 「就地展开对应原文段落」的交互形态。 */
const CITATION_PREVIEW_LEN = 150;

let _answerText = '';        // 已到达的 token 拼接结果（尚未经终帧校正）
let _answerNode = null;      // 承载正文的那个文本节点（见 appendToken 的说明）
let _answerEl = null;
let _citationEl = null;

function _answerBody() {
  if (!_answerEl) _answerEl = document.getElementById('answer-body');
  return _answerEl;
}

function _citationBody() {
  if (!_citationEl) _citationEl = document.getElementById('citation-body');
  return _citationEl;
}

/** 清空结果区，回到初始空态。每次提交前调用。 */
function resetResult() {
  _answerText = '';
  _answerNode = null;
  const answer = _answerBody();
  if (answer) {
    answer.textContent = '';
    answer.classList.remove('urgent');
    const ph = document.createElement('p');
    ph.className = 'placeholder';
    ph.textContent = '正在处理…';
    answer.appendChild(ph);
  }
  const cites = _citationBody();
  if (cites) {
    cites.textContent = '';
    const ph = document.createElement('p');
    ph.className = 'placeholder';
    ph.textContent = '暂无引用。';
    cites.appendChild(ph);
  }
}

/**
 * 追加一段正文。
 *
 * ⚠️ **追加到同一个文本节点**（用 `appendData`），而不是每次 `appendChild`
 * 一个新节点。两个原因：
 *
 * 1. 一次回答会来几百上千个 token。每次新建节点就是几百上千个 DOM 节点，
 *    而 `appendData` 只改一个已有节点的数据 —— 开销差一个量级。
 * 2. 它给出一个**可断言的不变量**："那个承载正文的节点，从头到尾是同一个
 *    对象"。这比"没有重建 DOM"这种模糊说法更容易测，也更容易在评审时看出来
 *    有没有被写坏。
 *
 * 与 FR-030 的关系：只往尾部加数据，永远不可能出现"先渲染后重排"，
 * 而重排正是 FR-030 要防的。
 *
 * @param {string} text
 */
function appendToken(text) {
  if (!text) return;
  const el = _answerBody();
  if (!el) return;

  if (_answerNode === null) {
    el.textContent = '';                          // 首次追加时清掉占位
    _answerNode = document.createTextNode('');
    el.appendChild(_answerNode);
  }

  _answerText += text;
  _answerNode.appendData(text);
}

/** 渲染引用片段。按传入顺序，**不排序**。 */
function renderCitations(citations) {
  const el = _citationBody();
  if (!el) return;

  el.textContent = '';

  if (!citations || citations.length === 0) {
    const ph = document.createElement('p');
    ph.className = 'placeholder';
    ph.textContent = '暂无引用。';
    el.appendChild(ph);
    return;
  }

  citations.forEach((c, i) => {
    el.appendChild(buildCitationNode(c, i));
  });
}

function buildCitationNode(c, index) {
  const wrap = document.createElement('div');
  wrap.className = 'citation-item';

  const head = document.createElement('div');
  head.className = 'citation-head';

  const idx = document.createElement('span');
  idx.className = 'citation-index';
  idx.textContent = String(index + 1);

  const file = document.createElement('span');
  file.className = 'citation-file';
  file.textContent = c.file_name;

  const page = document.createElement('span');
  page.className = 'citation-page';
  page.textContent =
    c.page_start === c.page_end
      ? `第 ${c.page_start} 页`
      : `第 ${c.page_start}–${c.page_end} 页`;

  head.append(idx, file, page);

  if (typeof c.score === 'number') {
    const score = document.createElement('span');
    score.className = 'citation-score';
    score.textContent = c.score.toFixed(2);
    head.appendChild(score);
  }

  const text = c.text || '';
  const preview = text.length > CITATION_PREVIEW_LEN
    ? text.slice(0, CITATION_PREVIEW_LEN) + '…'
    : text;

  const body = document.createElement('p');
  body.className = 'citation-text';
  // textContent：原文里的 < > 【】 一律原样显示，不作标记解释
  body.textContent = preview;

  wrap.append(head, body);

  if (text.length > CITATION_PREVIEW_LEN) {
    wrap.appendChild(buildToggle(body, text, preview));
  }

  return wrap;
}

/** 长引用的展开/收起。状态只在这一个闭包里，不需要全局变量。 */
function buildToggle(bodyEl, fullText, previewText) {
  let expanded = false;

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'citation-toggle';
  btn.textContent = '展开原文';

  btn.addEventListener('click', () => {
    expanded = !expanded;
    bodyEl.textContent = expanded ? fullText : previewText;
    btn.textContent = expanded ? '收起' : '展开原文';
  });

  return btn;
}

/**
 * 终帧到达。**以 answer_text 为准**，纠正已渲染内容。
 *
 * 为什么必须校正：流式下用户看到的是 token 拼起来的，而终帧的 answer_text 是
 * 服务端权威全文。二者有可能不一致（丢事件、拼接 bug）。docs/05 §3.1.5 要求
 * 前端直接渲染 answer_text 整体、不得自行拼接 —— 校正正是这条要求在流式形态下
 * 的实现方式：渐进渲染只是预览，最终显示的必须与服务端逐字一致。
 */
function renderDone(envelope) {
  const el = _answerBody();
  if (!el) return;

  const authoritative = envelope.answer_text || '';

  if (_answerText && _answerText !== authoritative) {
    // 不一致说明链路上有问题。不静默吞掉 —— 这是一个真实的信号，
    // 排查时它会指向具体某次提问。
    console.warn(
      '[医知源] 流式拼接结果与终帧 answer_text 不一致，已按终帧校正。',
      { streamed: _answerText, authoritative }
    );
  }

  el.textContent = authoritative;
  _answerText = authoritative;
  // 整体替换后，原文本节点已脱离文档，引用必须失效 —— 否则后续若再追加，
  // 会往一个不在页面上的节点里写，表现为"内容加不进去"。
  _answerNode = null;

  if (envelope.emergency && envelope.emergency.triggered) {
    el.classList.add('urgent');
  }

  if (Array.isArray(envelope.citations)) {
    renderCitations(envelope.citations);
  }
}

/** 在答案区显示一条错误说明，与"能力未就绪"的空态在视觉上可区分。 */
function showResultError(message) {
  const el = _answerBody();
  if (!el) return;
  el.textContent = '';
  el.classList.remove('urgent');
  _answerNode = null;   // 同 renderDone：整体替换后旧节点引用作废

  const p = document.createElement('p');
  p.className = 'placeholder';
  p.textContent = message;
  el.appendChild(p);
}
