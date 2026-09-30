"use client";

/**
 * 问答页：流式对话主界面。
 *
 * 状态机说明（对应后端 app/api/chat_stream.py 的事件序列）：
 *   idle → streaming（收到 message_start）
 *        → 逐 token 追加内容
 *        → 收集 citation
 *        → done（收到 message_end）
 *        → 或 error（收到 error 事件；后端此时不发 message_end，必须自行收尾）
 *
 * 关键约束：
 * 1. 用户问题在发请求前就插入列表，让界面立刻有反馈，不等首字节；
 * 2. 流式期间禁用输入与发送，避免同一会话并发请求打乱短期记忆；
 * 3. 支持中止（AbortController），中止后保留已生成内容而不是丢弃；
 * 4. 只有 error 事件才算失败 —— HTTP 200 但中途失败是后端的正常设计。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { RequireAuth } from "@/components/require-auth";
import { MessageItem, citationAnchorId, type ChatMessage } from "@/components/chat/message-item";
import { Button, Field, Notice, Tag, cx } from "@/components/ui/primitives";
import {
  createSessionId,
  deleteChatSession,
  fetchChatMessages,
  fetchChatSessions,
  mapSessionMessages,
  streamChat,
} from "@/lib/api-chat";
import { ApiError } from "@/lib/api-client";
import {
  DEFAULT_KNOWLEDGE_BASE_ID,
  type ChatStreamOptions,
  type DocumentType,
  type SessionListItem,
} from "@/lib/types";

/** 会话 ID 的本地存储 key；刷新后同一会话继续，短期记忆才有意义 */
const SESSION_STORAGE_KEY = "legal_rag_chat_session_id";

const DOCUMENT_TYPE_OPTIONS: Array<{ value: DocumentType; label: string }> = [
  { value: "law", label: "法律" },
  { value: "administrative_regulation", label: "行政法规" },
  { value: "judicial_interpretation", label: "司法解释" },
];

/** 预设提问：首期只覆盖劳动法，给几个高频场景降低上手成本 */
const SUGGESTED_QUESTIONS = [
  "经济补偿按什么标准计算？",
  "公司单方解除劳动合同需要满足哪些条件？",
  "试用期最长可以约定多久？",
  "加班工资的计算基数怎么确定？",
];

export default function ChatPage() {
  return (
    <RequireAuth>
      <ChatWorkspace />
    </RequireAuth>
  );
}

function ChatWorkspace() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(true);
  const [sessionError, setSessionError] = useState<string | null>(null);

  /** 检索与过滤选项 */
  const [asOfDate, setAsOfDate] = useState("");
  const [documentTypes, setDocumentTypes] = useState<DocumentType[]>([
    "law",
    "administrative_regulation",
    "judicial_interpretation",
  ]);
  const [enableQueryRewrite, setEnableQueryRewrite] = useState(true);
  const [showOptions, setShowOptions] = useState(false);

  /** 中止句柄：用户点「停止生成」或组件卸载时用 */
  const abortRef = useRef<AbortController | null>(null);
  /** 滚动锚点，每来一个 token 就滚到底部 */
  const bottomRef = useRef<HTMLDivElement | null>(null);
  /** 会话加载序号：只允许最后一次切换请求写入消息 */
  const sessionLoadVersionRef = useRef(0);

  // 首次进入页面读取服务端会话列表，并恢复上次打开的会话
  useEffect(() => {
    let cancelled = false;

    const loadSessions = async () => {
      setIsLoadingSessions(true);
      setSessionError(null);
      try {
        const loadedSessions = await fetchChatSessions();
        if (cancelled) {
          return;
        }
        setSessions(loadedSessions);
        const saved = window.localStorage.getItem(SESSION_STORAGE_KEY);
        const activeSession = loadedSessions.find(
          (item) => item.session_id === saved,
        ) ?? loadedSessions[0];
        if (activeSession) {
          setSessionId(activeSession.session_id);
          window.localStorage.setItem(
            SESSION_STORAGE_KEY,
            activeSession.session_id,
          );
          const history = await fetchChatMessages(activeSession.session_id);
          if (!cancelled) {
            setMessages(mapSessionMessages(history));
          }
        } else {
          const created = createSessionId();
          if (!cancelled) {
            setSessionId(created);
            window.localStorage.setItem(SESSION_STORAGE_KEY, created);
            setMessages([]);
          }
        }
      } catch (error) {
        if (!cancelled) {
          setSessionError(
            error instanceof ApiError
              ? error.message
              : "历史会话加载失败，请稍后重试",
          );
        }
      } finally {
        if (!cancelled) {
          setIsLoadingSessions(false);
        }
      }
    };

    void loadSessions();
    return () => {
      cancelled = true;
    };
  }, []);

  // 卸载时务必中止在途请求，否则组件已销毁仍在 setState
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  // 内容变化时滚到底部，让用户始终看到最新生成位置
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  /** 局部更新某条助手消息 */
  const patchMessage = useCallback(
    (messageId: string, patch: Partial<ChatMessage>) => {
      setMessages((previous) =>
        previous.map((item) =>
          item.id === messageId ? { ...item, ...patch } : item,
        ),
      );
    },
    [],
  );

  /** 局部更新某条助手消息的内容（函数式，避免流式追加丢失） */
  const appendMessageContent = useCallback(
    (messageId: string, text: string) => {
      setMessages((previous) =>
        previous.map((item) =>
          item.id === messageId
            ? { ...item, content: item.content + text }
            : item,
        ),
      );
    },
    [],
  );

  const handleSend = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || isStreaming || !sessionId) {
        return;
      }

      const userMessageId = `local_user_${Date.now()}`;
      const assistantMessageId = `local_assistant_${Date.now()}`;

      setMessages((previous) => [
        ...previous,
        {
          id: userMessageId,
          role: "user",
          content: trimmed,
          citations: [],
          isStreaming: false,
        },
        {
          id: assistantMessageId,
          role: "assistant",
          content: "",
          citations: [],
          isStreaming: true,
        },
      ]);
      setInput("");
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      const options: ChatStreamOptions = {
        top_k: 8,
        enable_query_rewrite: enableQueryRewrite,
        // 长期记忆接口尚未实现，明确传 false，避免让后端处理未完成的链路
        enable_long_term_memory: false,
        jurisdiction: "中国大陆",
        as_of_date: asOfDate || null,
        document_types: documentTypes,
      };

      try {
        await streamChat(
          {
            sessionId,
            message: trimmed,
            options,
            signal: controller.signal,
          },
          {
            onToken: (event) => {
              if (event.text) {
                appendMessageContent(assistantMessageId, event.text);
              }
            },
            onCitation: (event) => {
              setMessages((previous) =>
                previous.map((item) =>
                  item.id === assistantMessageId
                    ? { ...item, citations: [...item.citations, event] }
                    : item,
                ),
              );
            },
            onReplace: (event) => {
              // 真流式护栏（方案 A）：已流出的 token 被判定不可信，整段替换为后端给的安全文案
              setMessages((previous) =>
                previous.map((item) =>
                  item.id === assistantMessageId
                    ? { ...item, content: event.text }
                    : item,
                ),
              );
            },
            onMessageEnd: (event) => {
              patchMessage(assistantMessageId, {
                isStreaming: false,
                elapsedSeconds: event.elapsed_seconds,
              });
            },
            onError: (event) => {
              // 后端 error 事件后不再发 message_end，这里必须自行结束流式态
              patchMessage(assistantMessageId, {
                isStreaming: false,
                error: {
                  message: event.message || "生成失败，请稍后重试",
                  retryable: event.retryable,
                },
              });
            },
          },
        );
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          // 用户主动停止：保留已生成内容，只结束流式态
          patchMessage(assistantMessageId, { isStreaming: false });
        } else {
          patchMessage(assistantMessageId, {
            isStreaming: false,
            error: {
              message:
                error instanceof ApiError
                  ? error.message
                  : "请求失败，请稍后重试",
              retryable: true,
            },
          });
        }
      } finally {
        // 流正常结束时也要收尾：后端异常路径下 message_end 不会到达
        patchMessage(assistantMessageId, { isStreaming: false });
        setIsStreaming(false);
        abortRef.current = null;
        try {
          setSessions(await fetchChatSessions());
        } catch {
          // 问答已完成时，列表刷新失败不覆盖当前回答
        }
      }
    },
    [
      appendMessageContent,
      asOfDate,
      documentTypes,
      enableQueryRewrite,
      isStreaming,
      patchMessage,
      sessionId,
    ],
  );

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const handleSelectSession = useCallback(
    async (selectedSessionId: string) => {
      if (isStreaming || selectedSessionId === sessionId) {
        return;
      }
      const loadVersion = sessionLoadVersionRef.current + 1;
      sessionLoadVersionRef.current = loadVersion;
      setSessionError(null);
      setSessionId(selectedSessionId);
      setMessages([]);
      window.localStorage.setItem(SESSION_STORAGE_KEY, selectedSessionId);
      try {
        const history = await fetchChatMessages(selectedSessionId);
        if (sessionLoadVersionRef.current !== loadVersion) {
          return;
        }
        setMessages(mapSessionMessages(history));
      } catch (error) {
        if (sessionLoadVersionRef.current !== loadVersion) {
          return;
        }
        setSessionError(
          error instanceof ApiError ? error.message : "历史消息加载失败，请稍后重试",
        );
      }
    },
    [isStreaming, sessionId],
  );

  const handleNewSession = useCallback(() => {
    if (isStreaming) {
      return;
    }
    sessionLoadVersionRef.current += 1;
    const created = createSessionId();
    setSessionError(null);
    setSessionId(created);
    window.localStorage.setItem(SESSION_STORAGE_KEY, created);
    setMessages([]);
  }, [isStreaming]);

  const handleDeleteSession = useCallback(
    async (deletedSessionId: string) => {
      if (!window.confirm(`确定删除会话“${sessions.find((item) => item.session_id === deletedSessionId)?.title ?? "此会话"}”吗？`)) {
        return;
      }
      setSessionError(null);
      try {
        await deleteChatSession(deletedSessionId);
        setSessions((previous) =>
          previous.filter((item) => item.session_id !== deletedSessionId),
        );
        if (deletedSessionId === sessionId) {
          sessionLoadVersionRef.current += 1;
          const nextSessionId = createSessionId();
          setSessionId(nextSessionId);
          window.localStorage.setItem(SESSION_STORAGE_KEY, nextSessionId);
          setMessages([]);
        }
      } catch (error) {
        setSessionError(
          error instanceof ApiError ? error.message : "会话删除失败，请稍后重试",
        );
      }
    },
    [isStreaming, sessionId],
  );

  const handleCitationClick = useCallback((messageId: string, index: number) => {
    const target = document.getElementById(citationAnchorId(messageId, index));
    target?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, []);

  const toggleDocumentType = useCallback((value: DocumentType) => {
    setDocumentTypes((previous) =>
      previous.includes(value)
        ? previous.filter((item) => item !== value)
        : [...previous, value],
    );
  }, []);

  const isEmpty = messages.length === 0;
  const selectedTypeSummary = useMemo(() => {
    if (documentTypes.length === DOCUMENT_TYPE_OPTIONS.length) {
      return "全部";
    }
    if (documentTypes.length === 0) {
      return "未选择";
    }
    return documentTypes
      .map(
        (value) =>
          DOCUMENT_TYPE_OPTIONS.find((option) => option.value === value)?.label ??
          value,
      )
      .join("、");
  }, [documentTypes]);

  return (
    <div className="mx-auto flex w-full max-w-[1440px] flex-1 flex-col px-4 py-4 sm:px-6 sm:py-6 lg:flex-row lg:gap-6">
      <aside className="mb-4 shrink-0 lg:mb-0 lg:w-64">
        <div className="rounded-xl border border-ink-200 bg-white p-3 lg:sticky lg:top-6">
          <div className="mb-3 flex items-center justify-between px-1">
            <div>
              <h2 className="text-sm font-semibold text-ink-900">历史会话</h2>
              <p className="mt-0.5 text-xs text-ink-400">点击查看之前的对话</p>
            </div>
            <span className="rounded-full bg-ink-50 px-2 py-1 text-xs text-ink-500">
              {sessions.length}
            </span>
          </div>
          <div className="max-h-56 space-y-1.5 overflow-y-auto pr-1 lg:max-h-[calc(100vh-12rem)]">
            {isLoadingSessions ? (
              <p className="px-2 py-3 text-xs text-ink-500">正在加载会话…</p>
            ) : sessions.length === 0 ? (
              <p className="px-2 py-3 text-xs text-ink-500">暂无历史会话</p>
            ) : (
              sessions.map((item) => (
                <div
                  key={item.session_id}
                  className={cx(
                    "group flex items-stretch rounded-lg border transition-colors",
                    item.session_id === sessionId
                      ? "border-seal-300 bg-seal-50"
                      : "border-transparent hover:border-ink-200 hover:bg-ink-50",
                  )}
                >
                  <button
                    type="button"
                    disabled={isStreaming}
                    onClick={() => void handleSelectSession(item.session_id)}
                    className={cx(
                      "min-w-0 flex-1 px-3 py-2.5 text-left text-ink-700",
                      "disabled:cursor-not-allowed disabled:opacity-60",
                      item.session_id === sessionId ? "text-seal-800" : "",
                    )}
                  >
                    <span className="block truncate text-sm">{item.title}</span>
                    <span className="mt-1 block text-xs text-ink-400">
                      {item.message_count} 条消息
                    </span>
                  </button>
                  <button
                    type="button"
                    aria-label={`删除会话：${item.title}`}
                    disabled={isStreaming}
                    onClick={() => void handleDeleteSession(item.session_id)}
                    className="shrink-0 px-2 text-xs text-ink-400 opacity-100 transition-opacity hover:text-red-600 lg:opacity-0 lg:group-hover:opacity-100 disabled:cursor-not-allowed disabled:opacity-30"
                  >
                    删除
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        {sessionError ? <Notice tone="error">{sessionError}</Notice> : null}
        <div className="rule-line mb-5 flex flex-wrap items-center gap-3 border-b pb-4">
        <div className="flex items-baseline gap-2">
          <h1 className="font-serif text-lg text-ink-900">法律问答</h1>
          <span className="text-xs text-ink-500">
            知识库 {DEFAULT_KNOWLEDGE_BASE_ID} · 法域 中国大陆
          </span>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <Button
            variant="secondary"
            onClick={() => setShowOptions((previous) => !previous)}
            aria-expanded={showOptions}
            className="px-3 py-1.5 text-xs"
          >
            检索设置
          </Button>
          <Button
            variant="ghost"
            onClick={handleNewSession}
            disabled={isStreaming}
            className="px-3 py-1.5 text-xs"
          >
            新会话
          </Button>
        </div>
      </div>

      {showOptions ? (
        <div className="mb-5 rounded-lg border border-ink-200 bg-white p-4">
          <div className="grid gap-5 sm:grid-cols-2">
            <Field
              label="适用时间点"
              type="date"
              value={asOfDate}
              hint="按法条生效/失效日期筛选当时有效的版本；留空则不按时间过滤"
              onChange={(event) => setAsOfDate(event.target.value)}
            />

            <div className="flex flex-col gap-1.5">
              <span className="text-sm font-medium text-ink-700">文书类型</span>
              <div className="flex flex-wrap gap-2">
                {DOCUMENT_TYPE_OPTIONS.map((option) => {
                  const isSelected = documentTypes.includes(option.value);
                  return (
                    <button
                      key={option.value}
                      type="button"
                      aria-pressed={isSelected}
                      onClick={() => toggleDocumentType(option.value)}
                      className={cx(
                        "rounded border px-3 py-1.5 text-xs transition-colors duration-150",
                        isSelected
                          ? "border-seal-300 bg-seal-50 font-medium text-seal-700"
                          : "border-ink-200 bg-white text-ink-600 hover:border-ink-300",
                      )}
                    >
                      {option.label}
                    </button>
                  );
                })}
              </div>
              <span className="text-xs text-ink-500">
                已选：{selectedTypeSummary}
              </span>
            </div>
          </div>

          <label className="mt-4 flex cursor-pointer items-start gap-2.5">
            <input
              type="checkbox"
              checked={enableQueryRewrite}
              onChange={(event) => setEnableQueryRewrite(event.target.checked)}
              className="mt-0.5 h-4 w-4 accent-seal-600"
            />
            <span className="text-xs leading-relaxed text-ink-600">
              <span className="font-medium text-ink-800">启用指代补全</span>
              <br />
              把「那 12 期呢」这类省略主语的追问补全成完整问题再检索。
            </span>
          </label>
        </div>
      ) : null}

      <div className="flex-1 overflow-hidden">
        {isEmpty ? (
          <EmptyState
            onSelect={(question) => void handleSend(question)}
            disabled={isStreaming}
          />
        ) : (
          <div className="space-y-7 pb-4">
            {messages.map((message) => (
              <MessageItem
                key={message.id}
                message={message}
                onCitationClick={handleCitationClick}
              />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <div className="sticky bottom-0 mt-4 border-t border-ink-200 bg-paper pt-4">
        <form
          className="flex items-end gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            void handleSend(input);
          }}
        >
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              // Enter 发送、Shift+Enter 换行；输入法组合中的 Enter 不触发发送
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing
              ) {
                event.preventDefault();
                void handleSend(input);
              }
            }}
            rows={2}
            placeholder="描述你的劳动法问题，例如：经济补偿按什么标准计算？"
            disabled={isStreaming}
            className={cx(
              "min-h-[3.25rem] flex-1 resize-none rounded-md border border-ink-200 bg-white",
              "px-3.5 py-2.5 text-sm leading-relaxed text-ink-900",
              "placeholder:text-ink-400 focus:border-ink-500 focus:outline-none",
              "disabled:bg-ink-50 disabled:text-ink-500",
            )}
          />

          {isStreaming ? (
            <Button type="button" variant="secondary" onClick={handleStop}>
              停止生成
            </Button>
          ) : (
            <Button type="submit" disabled={!input.trim()}>
              发送
            </Button>
          )}
        </form>

        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-ink-500">
          <Tag tone="neutral">Enter 发送</Tag>
          <Tag tone="neutral">Shift + Enter 换行</Tag>
          {asOfDate ? <Tag tone="accent">适用时间 {asOfDate}</Tag> : null}
        </div>
      </div>
      </main>
    </div>
  );
}

/** 空态：说明能力边界 + 提供预设提问 */
function EmptyState({
  onSelect,
  disabled,
}: {
  onSelect: (question: string) => void;
  disabled: boolean;
}) {
  return (
    <div className="py-10">
      <div className="max-w-reading">
        <h2 className="font-serif text-xl text-ink-900">
          可以问什么
        </h2>
        <p className="prose-legal mt-3 text-sm text-ink-600">
          本助手只依据知识库中已收录的中国大陆全国层面劳动法法规与司法解释文本作答。
          每条结论都会标注引用编号，检索不到依据时会直接告知，不会用常识补充。
        </p>
      </div>

      <div className="mt-8">
        <p className="mb-3 text-xs font-medium uppercase tracking-wide text-ink-500">
          试试这些
        </p>
        <div className="grid gap-2.5 sm:grid-cols-2">
          {SUGGESTED_QUESTIONS.map((question) => (
            <button
              key={question}
              type="button"
              disabled={disabled}
              onClick={() => onSelect(question)}
              className={cx(
                "group rounded-md border border-ink-200 bg-white px-4 py-3 text-left",
                "transition-colors duration-150 hover:border-ink-400 hover:bg-ink-50",
                "disabled:cursor-not-allowed disabled:opacity-60",
              )}
            >
              <span className="text-sm text-ink-700 group-hover:text-ink-900">
                {question}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="mt-8 max-w-reading">
        <Notice tone="info">
          <strong className="font-medium">使用前须知</strong>
          <br />
          本系统提供的是法律信息整理，不是法律意见，也不代表执业律师。
          涉及具体纠纷或权益时，请携带完整材料咨询执业律师。
        </Notice>
      </div>
    </div>
  );
}
