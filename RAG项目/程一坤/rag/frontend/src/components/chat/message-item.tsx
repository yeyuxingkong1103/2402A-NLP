"use client";

/**
 * 单条消息的展示。
 *
 * 用户消息与助手消息采用两种视觉：用户靠右、印章红底；助手靠左、白底文书感，
 * 因为助手回答需要承载标题、列表、引用与法源卡片，必须给足横向空间。
 */

import { AnswerContent } from "@/components/chat/answer-content";
import { CitationCard } from "@/components/chat/citation-card";
import { LoadingDots, Notice } from "@/components/ui/primitives";
import type { CitationEvent } from "@/lib/types";

/** 一条聊天消息 */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: CitationEvent[];
  /** 是否仍在流式生成中 */
  isStreaming: boolean;
  /** 服务端从开始生成到回答完成的耗时；历史消息可能没有 */
  elapsedSeconds?: number;
  /** 本次回答发生的错误；有值时不再展示内容 */
  error?: { message: string; retryable: boolean } | null;
}

interface MessageItemProps {
  message: ChatMessage;
  /** 点击回答中的 [n] 编号时滚动到对应法源卡片 */
  onCitationClick: (messageId: string, index: number) => void;
}

export function MessageItem({ message, onCitationClick }: MessageItemProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-lg rounded-br-sm bg-seal-600 px-4 py-3">
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-white">
            {message.content}
          </p>
        </div>
      </div>
    );
  }

  const hasContent = message.content.length > 0;

  return (
    <div className="flex justify-start">
      <div className="w-full max-w-3xl">
        <div className="mb-1.5 flex items-center gap-2">
          <span className="font-serif text-xs tracking-wide text-ink-500">
            法律知识助手
          </span>
          {message.isStreaming ? (
            <span className="text-xs text-ink-400">正在生成</span>
          ) : message.elapsedSeconds !== undefined ? (
            <span className="text-xs text-ink-400">
              耗时 {message.elapsedSeconds.toFixed(2)} 秒
            </span>
          ) : null}
        </div>

        <div className="rounded-lg rounded-tl-sm border border-ink-200 bg-white px-5 py-4">
          {message.error ? (
            <Notice tone="error">
              {message.error.message}
              {message.error.retryable ? "（可重试）" : ""}
            </Notice>
          ) : null}

          {!message.error && !hasContent && message.isStreaming ? (
            <LoadingDots label="正在检索法条并组织回答" />
          ) : null}

          {hasContent ? (
            <div className={message.isStreaming ? "streaming-caret" : undefined}>
              <AnswerContent
                text={message.content}
                onCitationClick={(index) => onCitationClick(message.id, index)}
              />
            </div>
          ) : null}
        </div>

        {message.citations.length > 0 ? (
          <section className="mt-4">
            <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-ink-500">
              本次引用法源（{message.citations.length}）
            </h3>
            <div className="space-y-2">
              {message.citations.map((citation, index) => (
                <CitationCard
                  key={`${citation.chunk_id}-${index}`}
                  index={index + 1}
                  citation={citation}
                  defaultExpanded={message.citations.length === 1}
                  anchorId={citationAnchorId(message.id, index + 1)}
                />
              ))}
            </div>
          </section>
        ) : null}
      </div>
    </div>
  );
}

/** 引用卡片的 DOM id，供回答里的 [n] 编号定位 */
export function citationAnchorId(messageId: string, index: number): string {
  return `citation-${messageId}-${index}`;
}
