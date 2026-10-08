/* SSE 解析：POST + fetch + ReadableStream。
 *
 * 契约见 specs/006-medical-qa-input/contracts/sse.md。
 *
 * 为什么不用 EventSource：它只支持 GET，而问题必须放在请求体里。塞进 query
 * string 会同时带来三宗罪 —— 问题原文进服务器访问日志（隐私面扩大）、
 * 长度受 URL 上限约束、中文 URL 编码后可读性归零。
 *
 * ⚠️ 下面两处是**真陷阱**，不是防御性代码：
 *
 * 1. TextDecoder 必须带 { stream: true }。
 *    不设它，一个多字节 UTF-8 字符被 chunk 边界切断时会产生乱码。
 *    中文几乎必然触发 —— 换个英文问题可能一次都碰不到。
 *
 * 2. 按 \n\n 切块时必须**保留未完成的尾部**。
 *    chunk 边界由 TCP 决定，与我们的事件分隔符毫无关系。一个 data: 行被切成
 *    两次 read() 到达是**常态而非异常**。丢掉尾部会随机丢事件，且表现为
 *    "偶尔少一帧"，极难复现。
 */

const ASK_ENDPOINT = '/ask';

/**
 * 提交问题并消费 SSE 流。
 *
 * @param {string} question
 * @param {{onStatus?:Function, onCitations?:Function, onToken?:Function, onDone?:Function}} handlers
 * @returns {Promise<void>}
 */
async function streamAsk(question, handlers = {}) {
  let response;
  try {
    response = await fetch(ASK_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
  } catch (err) {
    throw streamError('network', '无法连接到服务，请确认服务已启动。');
  }

  // 请求**未被接受**。与"流中途断开"是两回事：前者服务端没开始处理，
  // 后者已经开始处理了。用户对这两种情况该做的事不同（改问题 vs 重试）。
  if (!response.ok) {
    throw streamError('rejected', await readErrorMessage(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let sawDone = false;

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // 保留最后一段（可能不完整），只处理已经以空行结束的完整事件。
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop();

      for (const block of blocks) {
        const evt = parseBlock(block);
        if (!evt) continue;
        dispatchEvent_(evt, handlers);
        if (evt.name === 'done') sawDone = true;
      }
    }

    // 收尾：流结束时缓冲区里可能还剩一个没有空行结尾的事件。
    const tail = parseBlock(buffer);
    if (tail) {
      dispatchEvent_(tail, handlers);
      if (tail.name === 'done') sawDone = true;
    }
  } catch (err) {
    if (err && err.kind) throw err;
    throw streamError('interrupted', '连接中断，回答可能不完整。');
  } finally {
    reader.cancel().catch(() => {});
  }

  if (!sawDone) {
    throw streamError('interrupted', '连接中断，回答可能不完整。');
  }
}

function parseBlock(block) {
  const text = block.trim();
  if (!text) return null;

  let name = 'message';
  const dataLines = [];

  for (const line of text.split('\n')) {
    if (line.startsWith(':')) continue;          // 注释行，忽略
    if (line.startsWith('event:')) {
      name = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (dataLines.length === 0) return null;

  let payload = {};
  try {
    payload = JSON.parse(dataLines.join('\n'));
  } catch (err) {
    console.warn('[医知源] 事件载荷不是合法 JSON，已忽略：', dataLines.join('\n'));
    return null;
  }

  return { name, data: payload };
}

function dispatchEvent_(evt, handlers) {
  switch (evt.name) {
    case 'status':
      if (handlers.onStatus) handlers.onStatus(evt.data);
      break;
    case 'citations':
      if (handlers.onCitations) handlers.onCitations(evt.data.citations || []);
      break;
    case 'token':
      if (handlers.onToken) handlers.onToken(evt.data.text || '');
      break;
    case 'done':
      if (handlers.onDone) handlers.onDone(evt.data);
      break;
    default:
      // 未知事件名静默忽略：后续新增事件类型不应打挂老前端。
      break;
  }
}

async function readErrorMessage(response) {
  try {
    const body = await response.json();
    if (body && body.error && body.error.message) return body.error.message;
  } catch (err) {
    /* 落到下面的兜底文案 */
  }
  return '请求未被接受，请稍后重试。';
}

function streamError(kind, message) {
  const err = new Error(message);
  err.kind = kind;   // 'network' | 'rejected' | 'interrupted'
  return err;
}
