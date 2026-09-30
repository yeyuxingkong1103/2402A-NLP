/**
 * SSE 解析器。
 *
 * 后端（app/chat/sse.py）按 `event: <name>\ndata: <json>\n\n` 分帧，
 * 且保证 token 片段拼接后与完整答案逐字一致。
 *
 * 关键实现点：网络分片不保证与事件边界对齐，一个事件可能被拆到两个 chunk，
 * 因此必须维护缓冲区、只按 `\n\n` 切分完整帧，绝不能对每个 chunk 直接 split。
 */

/** 一个已解析的 SSE 事件 */
export interface SseEvent {
  event: string;
  data: unknown;
}

/**
 * 把 SSE 字节流解析成事件序列。
 *
 * 用 TextDecoder 的 stream 模式而非一次性解码：
 * 中文一个字符占 3 字节，如果 chunk 正好切在字符中间，
 * 非流式解码会产生乱码替换符，答案里就会出现「?」。
 */
export async function* parseSseStream(
  response: Response,
): AsyncGenerator<SseEvent, void, void> {
  if (!response.body) {
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });

      let boundaryIndex = buffer.indexOf("\n\n");
      while (boundaryIndex !== -1) {
        const rawFrame = buffer.slice(0, boundaryIndex);
        buffer = buffer.slice(boundaryIndex + 2);
        const parsed = parseFrame(rawFrame);
        if (parsed) {
          yield parsed;
        }
        boundaryIndex = buffer.indexOf("\n\n");
      }
    }

    // 流结束时缓冲区可能残留最后一帧（后端正常以 \n\n 收尾，此处兜底）
    buffer += decoder.decode();
    const trailing = parseFrame(buffer);
    if (trailing) {
      yield trailing;
    }
  } finally {
    reader.releaseLock();
  }
}

/** 解析单个 SSE 帧；缺少 event 或 data 时返回 null 并跳过 */
function parseFrame(rawFrame: string): SseEvent | null {
  const frame = rawFrame.trim();
  if (!frame) {
    return null;
  }

  let eventName = "";
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (!eventName || dataLines.length === 0) {
    return null;
  }

  const rawData = dataLines.join("\n");
  try {
    return { event: eventName, data: JSON.parse(rawData) };
  } catch {
    // data 不是合法 JSON 说明帧损坏，丢弃而不是让整个流中断
    return null;
  }
}
