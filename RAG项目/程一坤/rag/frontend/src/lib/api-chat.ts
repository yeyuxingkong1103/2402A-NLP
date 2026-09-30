/**
 * 问答流式接口（app/api/chat.py）。
 *
 * 后端事件序列固定为：
 *   message_start → token × N（真流式，边生成边发）→ citation × M → message_end
 * 护栏拦截时（方案 A）：
 *   message_start → token × N → replace（整段替换全文）→ citation × M → message_end
 * 异常时：
 *   message_start → error（且不发 message_end）
 *
 * 上层（聊天页）按事件分发即可，不需要理解 SSE 分帧细节。
 */

import type { ChatMessage } from "@/components/chat/message-item";
import { requestJson, requestStream } from "@/lib/api-client";
import { parseSseStream, type SseEvent } from "@/lib/sse-parser";
import {
  DEFAULT_CHARACTER_ID,
  type ChatStreamOptions,
  type CitationEvent,
  type MessageEndEvent,
  type MessageStartEvent,
  type ReplaceEvent,
  type SessionListItem,
  type SessionMessage,
  type StreamErrorEvent,
  type TokenEvent,
} from "@/lib/types";

/** 流式问答的回调集合；每个事件对应一个可选回调 */
export interface ChatStreamHandlers {
  onMessageStart?: (event: MessageStartEvent) => void;
  onToken?: (event: TokenEvent) => void;
  onCitation?: (event: CitationEvent) => void;
  /** replace 事件：已流出的 token 被判定不可信，用 text 整段替换当前回答 */
  onReplace?: (event: ReplaceEvent) => void;
  onMessageEnd?: (event: MessageEndEvent) => void;
  /** error 事件：后端已给出可读文案与是否可重试 */
  onError?: (event: StreamErrorEvent) => void;
}

/** 一次流式问答的入参 */
export interface ChatStreamParams {
  sessionId: string;
  message: string;
  options?: ChatStreamOptions;
  signal?: AbortSignal;
}

/** 发起一次流式问答，把事件逐个交给回调 */
export async function streamChat(
  params: ChatStreamParams,
  handlers: ChatStreamHandlers,
): Promise<void> {
  const response = await requestStream("/chat/stream", {
    body: {
      session_id: params.sessionId,
      character_id: DEFAULT_CHARACTER_ID,
      message: params.message,
      options: params.options,
    },
    signal: params.signal,
  });

  for await (const sseEvent of parseSseStream(response)) {
    dispatchEvent(sseEvent, handlers);
  }
}

/** 按事件名分发；未知事件直接忽略，保证后端新增事件不会让前端崩 */
function dispatchEvent(sseEvent: SseEvent, handlers: ChatStreamHandlers): void {
  switch (sseEvent.event) {
    case "message_start":
      handlers.onMessageStart?.(sseEvent.data as MessageStartEvent);
      break;
    case "token":
      handlers.onToken?.(sseEvent.data as TokenEvent);
      break;
    case "citation":
      handlers.onCitation?.(sseEvent.data as CitationEvent);
      break;
    case "replace":
      handlers.onReplace?.(sseEvent.data as ReplaceEvent);
      break;
    case "message_end":
      handlers.onMessageEnd?.(sseEvent.data as MessageEndEvent);
      break;
    case "error":
      handlers.onError?.(sseEvent.data as StreamErrorEvent);
      break;
    default:
      break;
  }
}

/** 获取当前用户的历史会话列表。 */
export async function fetchChatSessions(): Promise<SessionListItem[]> {
  const result = await requestJson<{ items: SessionListItem[] }>("/sessions", {
    method: "GET",
  });
  return result.items;
}

/** 获取指定会话的历史消息。 */
export async function fetchChatMessages(
  sessionId: string,
): Promise<SessionMessage[]> {
  const result = await requestJson<{ items: SessionMessage[] }>(
    `/sessions/${encodeURIComponent(sessionId)}/messages`,
    { method: "GET" },
  );
  return result.items;
}

/** 删除当前用户的指定会话。 */
export function deleteChatSession(sessionId: string): Promise<{ session_id: string; status: string }> {
  return requestJson<{ session_id: string; status: string }>(
    `/sessions/${encodeURIComponent(sessionId)}`,
    { method: "DELETE" },
  );
}

/** 把后端历史消息转换为聊天组件使用的消息结构。 */
export function mapSessionMessages(messages: SessionMessage[]): ChatMessage[] {
  return messages.map((message) => ({
    id: message.message_id ?? `history_${message.created_at}`,
    role: message.role,
    content: message.content,
    citations: message.citations ?? [],
    isStreaming: false,
    elapsedSeconds:
      typeof message.elapsed_seconds === "number"
        ? message.elapsed_seconds
        : undefined,
  }));
}

/**
 * 生成会话 ID。
 *
 * 后端目前没有创建会话接口（docs/接口文档.md 7.1 的 POST /api/v1/sessions
 * 尚无实现），但 chat/stream 要求 session_id 必填，且它同时用作
 * Redis 短期记忆的 key 组成部分，因此由前端生成并持久化。
 *
 * 注意：后端把 session_id 用于短期记忆读写（app/api/chat_stream.py 中
 * enable_query_rewrite 为真时才传），所以同一会话刷新页面后必须沿用同一 ID，
 * 否则指代补全会失效 —— 这也是页面把它写进 localStorage 的原因。
 */
export function createSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `session_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`;
  }
  return `session_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}
