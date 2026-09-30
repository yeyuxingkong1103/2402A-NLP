"use client";

/**
 * 法律检索调试页。
 *
 * 对应 POST /api/v1/legal/search（app/api/legal_search.py）——
 * 该接口不调用大模型，专门用于观察检索链路本身，因此这里把
 * 「改写结果、两路召回条数、融合/重排分数、命中来源、生效状态」全部摊开，
 * 让检索质量问题能被直接看出来，而不是被大模型的措辞掩盖。
 */

import { useCallback, useState } from "react";

import { RequireAuth } from "@/components/require-auth";
import { Button, Field, LoadingDots, Notice, Tag, cx } from "@/components/ui/primitives";
import { ApiError } from "@/lib/api-client";
import {
  DOCUMENT_TYPE_LABELS,
  SCORE_SOURCE_LABELS,
  describeValidity,
  formatArticleLocation,
  searchLegalDocuments,
} from "@/lib/api-legal-search";
import {
  DEFAULT_KNOWLEDGE_BASE_ID,
  DEFAULT_LEGAL_DOMAIN,
  type DocumentType,
  type LegalSearchData,
  type LegalSearchResultItem,
} from "@/lib/types";

const DOCUMENT_TYPE_OPTIONS: Array<{ value: DocumentType; label: string }> = [
  { value: "law", label: "法律" },
  { value: "administrative_regulation", label: "行政法规" },
  { value: "judicial_interpretation", label: "司法解释" },
];

export default function LegalSearchPage() {
  return (
    <RequireAuth>
      <SearchWorkspace />
    </RequireAuth>
  );
}

function SearchWorkspace() {
  const [query, setQuery] = useState("");
  const [asOfDate, setAsOfDate] = useState("");
  const [topK, setTopK] = useState(10);
  const [documentTypes, setDocumentTypes] = useState<DocumentType[]>([
    "law",
    "administrative_regulation",
    "judicial_interpretation",
  ]);
  const [enableHybridSearch, setEnableHybridSearch] = useState(true);
  const [enableRerank, setEnableRerank] = useState(true);

  const [result, setResult] = useState<LegalSearchData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const trimmed = query.trim();
      if (!trimmed) {
        setError("请输入检索内容");
        return;
      }

      setIsLoading(true);
      setError(null);
      try {
        const data = await searchLegalDocuments({
          query: trimmed,
          knowledge_base_ids: [DEFAULT_KNOWLEDGE_BASE_ID],
          jurisdiction: "中国大陆",
          as_of_date: asOfDate || null,
          legal_domain: DEFAULT_LEGAL_DOMAIN,
          document_types: documentTypes,
          top_k: topK,
          enable_hybrid_search: enableHybridSearch,
          enable_rerank: enableRerank,
        });
        setResult(data);
      } catch (caught) {
        setResult(null);
        setError(
          caught instanceof ApiError ? caught.message : "检索失败，请稍后重试",
        );
      } finally {
        setIsLoading(false);
      }
    },
    [
      asOfDate,
      documentTypes,
      enableHybridSearch,
      enableRerank,
      query,
      topK,
    ],
  );

  const toggleDocumentType = useCallback((value: DocumentType) => {
    setDocumentTypes((previous) =>
      previous.includes(value)
        ? previous.filter((item) => item !== value)
        : [...previous, value],
    );
  }, []);

  return (
    <div className="mx-auto w-full max-w-shell flex-1 px-6 py-6">
      <div className="rule-line mb-5 border-b pb-4">
        <h1 className="font-serif text-lg text-ink-900">检索调试</h1>
        <p className="mt-1 text-xs text-ink-500">
          直接调用检索链路，不经过大模型。用于验证法条召回质量、时效过滤与分数分布。
        </p>
      </div>

      <form onSubmit={handleSearch} className="rounded-lg border border-ink-200 bg-white p-5">
        <Field
          label="检索内容"
          value={query}
          placeholder="例如：经济补偿 计算标准"
          onChange={(event) => setQuery(event.target.value)}
        />

        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          <Field
            label="适用时间点"
            type="date"
            value={asOfDate}
            hint="留空则不过滤时效"
            onChange={(event) => setAsOfDate(event.target.value)}
          />

          <Field
            label="返回条数 top_k"
            type="number"
            min={1}
            max={50}
            value={topK}
            hint="1–50"
            onChange={(event) => {
              const parsed = Number(event.target.value);
              setTopK(Number.isNaN(parsed) ? 10 : Math.min(50, Math.max(1, parsed)));
            }}
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
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-5">
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={enableHybridSearch}
              onChange={(event) => setEnableHybridSearch(event.target.checked)}
              className="h-4 w-4 accent-seal-600"
            />
            <span className="text-xs text-ink-600">启用混合检索（向量 + 关键词）</span>
          </label>

          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={enableRerank}
              onChange={(event) => setEnableRerank(event.target.checked)}
              className="h-4 w-4 accent-seal-600"
            />
            <span className="text-xs text-ink-600">启用重排</span>
          </label>

          <Button type="submit" disabled={isLoading} className="ml-auto">
            {isLoading ? "检索中…" : "执行检索"}
          </Button>
        </div>
      </form>

      <div className="mt-6 space-y-6">
        {error ? <Notice tone="error">{error}</Notice> : null}

        {isLoading ? (
          <div className="py-10 text-center">
            <LoadingDots label="正在检索法条" />
          </div>
        ) : null}

        {result && !isLoading ? <ResultPanel data={result} /> : null}
      </div>
    </div>
  );
}

function ResultPanel({ data }: { data: LegalSearchData }) {
  const { stats, rewrite, results } = data;

  return (
    <>
      <section className="rounded-lg border border-ink-200 bg-white p-5">
        <h2 className="font-serif text-base text-ink-900">链路统计</h2>

        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard label="向量召回" value={stats.vector_recall_count} />
          <StatCard label="关键词召回" value={stats.keyword_recall_count} />
          <StatCard label="融合后" value={stats.fused_count} />
          <StatCard label="重排保留" value={stats.reranked_count} />
        </div>

        {rewrite.changed ? (
          <div className="mt-4 rounded-md border border-ink-200 bg-ink-50 p-3.5">
            <p className="text-xs font-medium text-ink-700">查询改写生效</p>
            <p className="mt-1.5 text-xs text-ink-600">
              <span className="text-ink-400">原始：</span>
              {rewrite.original_query}
            </p>
            <p className="mt-1 text-xs text-ink-600">
              <span className="text-ink-400">改写：</span>
              {rewrite.rewritten_query}
            </p>
            {rewrite.reasons.length > 0 ? (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {rewrite.reasons.map((reason) => (
                  <Tag key={reason} tone="neutral">
                    {reason}
                  </Tag>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        <dl className="mt-4 grid gap-2 border-t border-ink-100 pt-3.5 text-xs sm:grid-cols-2">
          <FilterRow term="法域" value={data.filters.jurisdiction} />
          <FilterRow
            term="适用时间点"
            value={data.filters.as_of_date ?? "未指定（不按时效过滤）"}
          />
          <FilterRow
            term="文书类型"
            value={
              data.filters.document_types.length > 0
                ? data.filters.document_types
                    .map((item) => DOCUMENT_TYPE_LABELS[item] ?? item)
                    .join("、")
                : "未指定（不过滤）"
            }
          />
          <FilterRow
            term="法律领域"
            value={data.filters.legal_domain ?? "未指定"}
          />
        </dl>
      </section>

      <section>
        <h2 className="mb-3 font-serif text-base text-ink-900">
          命中条文（{results.length}）
        </h2>

        {results.length === 0 ? (
          <Notice tone="info">
            没有命中任何条文。若已设置适用时间点或文书类型，可尝试放宽条件后重试。
          </Notice>
        ) : (
          <div className="space-y-3">
            {results.map((item, index) => (
              <ResultCard
                key={item.chunk_id}
                rank={index + 1}
                item={item}
                asOfDate={data.filters.as_of_date}
              />
            ))}
          </div>
        )}
      </section>
    </>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md bg-ink-50 px-3.5 py-3">
      <p className="text-xs text-ink-500">{label}</p>
      <p className="mt-1 text-xl font-medium text-ink-900">{value}</p>
    </div>
  );
}

function FilterRow({
  term,
  value,
}: {
  term: string;
  /** 允许传 ReactNode：来源地址需要渲染成可点击的外链，不能只传字符串 */
  value: React.ReactNode;
}) {
  return (
    <div className="flex gap-3">
      <dt className="w-20 shrink-0 text-ink-500">{term}</dt>
      <dd className="min-w-0 flex-1 text-ink-700">{value}</dd>
    </div>
  );
}

function ResultCard({
  rank,
  item,
  asOfDate,
}: {
  rank: number;
  item: LegalSearchResultItem;
  asOfDate: string | null;
}) {
  const [isExpanded, setIsExpanded] = useState(rank <= 2);
  const validity = describeValidity(item);

  return (
    <article className="rounded-lg border border-ink-200 bg-white">
      <div className="flex items-start gap-3 px-4 py-3.5">
        <span className="mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded bg-ink-100 text-xs font-medium text-ink-700">
          {rank}
        </span>

        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-medium leading-snug text-ink-900">
            《{item.law_name}》{formatArticleLocation(item)}
          </h3>

          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            {item.document_type ? (
              <Tag tone="neutral">
                {DOCUMENT_TYPE_LABELS[item.document_type] ?? item.document_type}
              </Tag>
            ) : (
              <Tag tone="unknown">文书类型未知</Tag>
            )}
            <Tag tone={validity.tone === "current" ? "current" : "unknown"}>
              {validity.label}
            </Tag>
            {item.score_sources.map((source) => (
              <Tag key={source} tone="accent">
                {SCORE_SOURCE_LABELS[source] ?? source}
              </Tag>
            ))}
          </div>

          <div className="mt-2.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-500">
            <span>
              召回分 <span className="text-ink-700">{item.retrieval_score.toFixed(3)}</span>
            </span>
            {item.vector_score !== null ? (
              <span>
                向量 <span className="text-ink-700">{item.vector_score.toFixed(3)}</span>
              </span>
            ) : null}
            {item.keyword_score !== null ? (
              <span>
                BM25 <span className="text-ink-700">{item.keyword_score.toFixed(3)}</span>
              </span>
            ) : null}
            {item.fusion_score !== null ? (
              <span>
                RRF <span className="text-ink-700">{item.fusion_score.toFixed(4)}</span>
              </span>
            ) : null}
            {item.rerank_score !== null ? (
              <span>
                重排 <span className="text-ink-700">{item.rerank_score.toFixed(3)}</span>
              </span>
            ) : null}
          </div>
        </div>

        <Button
          variant="ghost"
          onClick={() => setIsExpanded((previous) => !previous)}
          aria-expanded={isExpanded}
          className="shrink-0 px-2 py-1 text-xs"
        >
          {isExpanded ? "收起" : "展开原文"}
        </Button>
      </div>

      {isExpanded ? (
        <div className="border-t border-ink-100 px-4 py-3.5">
          <p className="prose-legal whitespace-pre-wrap font-serif text-sm text-ink-800">
            {item.content}
          </p>

          <dl className="mt-3.5 space-y-1.5 border-t border-ink-100 pt-3.5 text-xs">
            <FilterRow term="文书编号" value={item.document_id ?? "无"} />
            <FilterRow term="条号" value={item.article_number ?? "无"} />
            <FilterRow term="款号" value={item.paragraph_number ?? "无"} />
            <FilterRow term="项号" value={item.item_number ?? "无"} />
            <FilterRow term="法域" value={item.jurisdiction ?? "未标注"} />
            <FilterRow term="发布机关" value={item.issuing_authority ?? "未标注"} />
            <FilterRow term="生效日期" value={item.effective_date ?? "未标注"} />
            <FilterRow term="失效日期" value={item.repeal_date ?? "未标注"} />
            <FilterRow
              term="时效命中"
              value={
                asOfDate
                  ? `按 ${asOfDate} 过滤，本条满足生效 ≤ ${asOfDate} < 失效`
                  : "未按时间点过滤"
              }
            />
            <FilterRow
              term="来源"
              value={
                item.source ? (
                  <a
                    href={item.source}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="break-all text-seal-600 underline-offset-2 hover:underline"
                  >
                    {item.source}
                  </a>
                ) : (
                  "未标注"
                )
              }
            />
          </dl>
        </div>
      ) : null}
    </article>
  );
}
