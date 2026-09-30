/**
 * 引用卡片。
 *
 * 后端 citation 事件只给 5 个字段（chunk_id / law_name / article_number /
 * paragraph_number / page，见 app/chat/service.py）。要展示法条原文和生效状态，
 * 就用 chunk_id 回查一次 /api/v1/legal/search —— 用 chunk_id 当查询词，
 * 关键词检索能精确命中同一条文，从而补齐原文与时效字段。
 *
 * 这是有意为之的设计：宁可多一次请求，也不把残缺信息包装成完整引用。
 */

"use client";

import { useCallback, useState } from "react";

import { Button, LoadingDots, Tag, cx } from "@/components/ui/primitives";
import { searchLegalDocuments, describeValidity, formatArticleLocation } from "@/lib/api-legal-search";
import { ApiError } from "@/lib/api-client";
import { DEFAULT_KNOWLEDGE_BASE_ID, type CitationEvent } from "@/lib/types";

/** 把条号渲染成中文数字形式，与法条原文习惯一致 */
const CHINESE_DIGITS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];

/** 阿拉伯数字条号转中文，如 47 → 四十七；非纯数字原样返回 */
export function toChineseArticleNumber(value: string): string {
  if (!/^\d+$/.test(value)) {
    return value;
  }
  const number = Number(value);
  if (number === 0) {
    return "零";
  }
  if (number < 10) {
    return CHINESE_DIGITS[number];
  }
  if (number < 20) {
    return number === 10 ? "十" : `十${CHINESE_DIGITS[number % 10]}`;
  }
  const tens = Math.floor(number / 10);
  const ones = number % 10;
  const tensText = `${CHINESE_DIGITS[tens]}十`;
  return ones === 0 ? tensText : `${tensText}${CHINESE_DIGITS[ones]}`;
}

/** 引用条目的展示标题，如「《劳动合同法》第四十七条」 */
export function formatCitationTitle(citation: {
  law_name: string;
  article_number: string | null;
  paragraph_number: string | null;
}): string {
  const articleText = citation.article_number
    ? `第${toChineseArticleNumber(citation.article_number)}条`
    : "";
  const paragraphText = citation.paragraph_number
    ? `第${toChineseArticleNumber(citation.paragraph_number)}款`
    : "";
  return `《${citation.law_name}》${articleText}${paragraphText}`;
}

interface CitationCardProps {
  index: number;
  citation: CitationEvent;
  /** 是否默认展开；首条通常展开，其余折叠 */
  defaultExpanded?: boolean;
  /** 用于滚动定位 */
  anchorId?: string;
}

/** 单条引用：折叠态只显示标题，展开后按需拉取原文 */
export function CitationCard({
  index,
  citation,
  defaultExpanded = true,
  anchorId,
}: CitationCardProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const [isLoading, setIsLoading] = useState(false);
  const [detail, setDetail] = useState<{
    content: string;
    meta: {
      document_type: string | null;
      is_current: boolean | null;
      effective_date: string | null;
      repeal_date: string | null;
      issuing_authority: string | null;
      source: string;
    };
  } | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const loadDetail = useCallback(async () => {
    setIsLoading(true);
    setLoadError(null);
    try {
      // 用条号 + 法规名当检索词，比用 chunk_id 更稳：chunk_id 不在关键词索引里
      const query = citation.article_number
        ? `${citation.law_name} 第${toChineseArticleNumber(citation.article_number)}条`
        : citation.law_name;
      const data = await searchLegalDocuments({
        query,
        knowledge_base_ids: [DEFAULT_KNOWLEDGE_BASE_ID],
        top_k: 5,
        enable_hybrid_search: true,
        enable_rerank: false,
      });
      // 优先取 chunk_id 完全一致的那条，避免展示成同一法规的另一条
      const matched =
        data.results.find((item) => item.chunk_id === citation.chunk_id) ??
        data.results[0];
      if (!matched) {
        setLoadError("未能取到该条文的原文");
        return;
      }
      setDetail({
        content: matched.content,
        meta: {
          document_type: matched.document_type,
          is_current: matched.is_current,
          effective_date: matched.effective_date,
          repeal_date: matched.repeal_date,
          issuing_authority: matched.issuing_authority,
          source: matched.source,
        },
      });
    } catch (error) {
      setLoadError(
        error instanceof ApiError ? error.message : "原文加载失败，请稍后重试",
      );
    } finally {
      setIsLoading(false);
    }
  }, [citation]);

  const handleToggle = useCallback(() => {
    setIsExpanded((previous) => {
      const next = !previous;
      // 只在首次展开时请求，重复展开不重复打接口
      if (next && !detail && !isLoading && !loadError) {
        void loadDetail();
      }
      return next;
    });
  }, [detail, isLoading, loadError, loadDetail]);

  const validity = detail
    ? describeValidity({
        is_current: detail.meta.is_current,
        effective_date: detail.meta.effective_date,
        repeal_date: detail.meta.repeal_date,
      })
    : null;

  return (
    <div
      id={anchorId}
      className={cx(
        "scroll-mt-20 rounded-md border bg-white transition-colors duration-150",
        isExpanded ? "border-ink-300" : "border-ink-200 hover:border-ink-300",
      )}
    >
      <div className="flex items-start gap-3 px-4 py-3">
        <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded bg-ink-100 text-xs font-medium text-ink-700">
          {index}
        </span>

        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium leading-snug text-ink-900">
            {formatCitationTitle(citation)}
          </p>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            {detail?.meta.document_type ? (
              <Tag tone="neutral">{detail.meta.document_type}</Tag>
            ) : null}
            {validity ? (
              <Tag tone={validity.tone === "current" ? "current" : "unknown"}>
                {validity.label}
              </Tag>
            ) : null}
            {citation.page ? (
              <span className="text-xs text-ink-500">第 {citation.page} 页</span>
            ) : null}
          </div>
        </div>

        <Button
          variant="ghost"
          onClick={handleToggle}
          aria-expanded={isExpanded}
          className="shrink-0 px-2 py-1 text-xs"
        >
          {isExpanded ? "收起" : "查看原文"}
        </Button>
      </div>

      {isExpanded ? (
        <div className="border-t border-ink-100 px-4 py-3">
          {isLoading ? <LoadingDots label="正在取回法条原文" /> : null}

          {loadError ? (
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs text-seal-600">{loadError}</p>
              <Button
                variant="secondary"
                onClick={() => void loadDetail()}
                className="shrink-0 px-2 py-1 text-xs"
              >
                重试
              </Button>
            </div>
          ) : null}

          {detail ? (
            <>
              <p className="prose-legal whitespace-pre-wrap font-serif text-sm text-ink-800">
                {detail.content}
              </p>

              <dl className="mt-3 space-y-1 border-t border-ink-100 pt-3 text-xs">
                <MetaRow
                  term="条文定位"
                  value={formatArticleLocation({
                    article_number: citation.article_number,
                    paragraph_number: citation.paragraph_number,
                    item_number: null,
                  })}
                />
                {detail.meta.issuing_authority ? (
                  <MetaRow term="发布机关" value={detail.meta.issuing_authority} />
                ) : null}
                {detail.meta.effective_date ? (
                  <MetaRow term="生效日期" value={detail.meta.effective_date} />
                ) : null}
                {detail.meta.source ? (
                  <MetaRow
                    term="来源"
                    value={
                      <a
                        href={detail.meta.source}
                        target="_blank"
                        rel="noreferrer noopener"
                        className="break-all text-seal-600 underline-offset-2 hover:underline"
                      >
                        {detail.meta.source}
                      </a>
                    }
                  />
                ) : null}
              </dl>
            </>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function MetaRow({ term, value }: { term: string; value: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <dt className="w-16 shrink-0 text-ink-500">{term}</dt>
      <dd className="min-w-0 flex-1 text-ink-700">{value}</dd>
    </div>
  );
}
