"use client";

/**
 * 管理端文档审核页（阶段 7.3）。
 *
 * 只调用后端三个既有接口（docs/接口文档.md）：
 * - GET  /admin/documents          6.3 列表（状态筛选 + 分页）
 * - GET  /admin/documents/{id}/detail  6.5 详情（元数据 + 分块预览 + 留痕）
 * - POST /admin/documents/{id}/review  6.4 提交审核决定
 *
 * 前端不做业务判断：能否审核、审核后状态如何流转都以后端为准，
 * 这里只负责展示与提交。驳回意见必填是后端留痕要求的界面约束，
 * 提交前在本页拦截空值，绝不给后端传空意见。
 *
 * 权限：403（40300）时显示明确的无权限提示与返回入口，不白屏；
 * 401（40100）由 api-client 清令牌、RequireAuth 踢回登录页。
 */

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { RequireAuth } from "@/components/require-auth";
import { Button, Notice, Tag, cx } from "@/components/ui/primitives";
import {
  getReviewDocumentDetail,
  listReviewDocuments,
  submitReviewDecision,
} from "@/lib/api-admin";
import { ApiError } from "@/lib/api-client";
import type {
  AdminDocumentDetailData,
  AdminReviewListItem,
  AdminReviewListData,
  ReviewDecisionResultData,
  ReviewStatus,
} from "@/lib/types";

const STATUS_OPTIONS: Array<{ value: ReviewStatus | ""; label: string }> = [
  { value: "pending_review", label: "待审核" },
  { value: "approved", label: "已通过" },
  { value: "rejected", label: "已驳回" },
  { value: "", label: "全部" },
];

const STATUS_TAGS: Record<
  ReviewStatus,
  { label: string; tone: "accent" | "current" | "expired" }
> = {
  pending_review: { label: "待审核", tone: "accent" },
  approved: { label: "已通过", tone: "current" },
  rejected: { label: "已驳回", tone: "expired" },
};

const PAGE_SIZE = 20;

export default function AdminDocumentsPage() {
  return (
    <RequireAuth>
      <AdminDocumentsWorkspace />
    </RequireAuth>
  );
}

function AdminDocumentsWorkspace() {
  const [status, setStatus] = useState<ReviewStatus | "">("pending_review");
  const [page, setPage] = useState(1);
  const [list, setList] = useState<AdminReviewListData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [forbidden, setForbidden] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [detail, setDetail] = useState<AdminDocumentDetailData | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  // 正在填写审核意见的行：{ document_id, decision }；null 表示收起
  const [reviewing, setReviewing] = useState<{
    documentId: number;
    decision: "approve" | "reject";
  } | null>(null);
  const [reviewNote, setReviewNote] = useState("");
  const [reviewNoteError, setReviewNoteError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [lastResult, setLastResult] = useState<ReviewDecisionResultData | null>(
    null,
  );

  const loadList = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setForbidden(false);
    try {
      const data = await listReviewDocuments({
        status: status === "" ? undefined : status,
        page,
        page_size: PAGE_SIZE,
      });
      setList(data);
    } catch (caught) {
      setList(null);
      if (caught instanceof ApiError && caught.code === 40300) {
        setForbidden(true);
      } else {
        setError(
          caught instanceof ApiError ? caught.message : "加载审核列表失败",
        );
      }
    } finally {
      setIsLoading(false);
    }
  }, [page, status]);

  useEffect(() => {
    void loadList();
  }, [loadList]);

  const openDetail = useCallback(async (documentId: number) => {
    setDetailError(null);
    setDetailLoading(true);
    try {
      setDetail(await getReviewDocumentDetail(documentId));
    } catch (caught) {
      setDetail(null);
      setDetailError(
        caught instanceof ApiError ? caught.message : "加载详情失败",
      );
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const startReview = useCallback(
    (documentId: number, decision: "approve" | "reject") => {
      setReviewing({ documentId, decision });
      setReviewNote("");
      setReviewNoteError(null);
      setLastResult(null);
    },
    [],
  );

  const submitReview = useCallback(async () => {
    if (!reviewing) return;
    const trimmed = reviewNote.trim();
    if (reviewing.decision === "reject" && !trimmed) {
      // 驳回意见必填：前端不给后端传空值（后端留痕要求）
      setReviewNoteError("驳回必须填写审核意见（留痕要求）");
      return;
    }
    setSubmitting(true);
    setReviewNoteError(null);
    try {
      const result = await submitReviewDecision(reviewing.documentId, {
        decision: reviewing.decision,
        review_note: trimmed || null,
      });
      setLastResult(result);
      setReviewing(null);
      // 列表与详情即时刷新
      await loadList();
      if (detail && detail.document_id === reviewing.documentId) {
        void openDetail(reviewing.documentId);
      }
    } catch (caught) {
      setReviewNoteError(
        caught instanceof ApiError ? caught.message : "提交审核失败，请稍后重试",
      );
    } finally {
      setSubmitting(false);
    }
  }, [detail, loadList, openDetail, reviewNote, reviewing]);

  const totalPages = list ? Math.max(1, Math.ceil(list.total / PAGE_SIZE)) : 1;

  if (forbidden) {
    return (
      <div className="mx-auto w-full max-w-shell flex-1 px-6 py-6">
        <div className="rounded-lg border border-ink-200 bg-white p-10 text-center">
          <p className="font-serif text-lg text-ink-900">无权限访问管理端</p>
          <p className="mt-2 text-sm text-ink-600">
            文档审核仅对管理员开放。如需权限，请联系系统管理员为当前账号授权。
          </p>
          <Link
            href="/chat"
            className="mt-6 inline-block rounded-md bg-seal-600 px-4 py-2.5 text-sm font-medium text-white transition-colors duration-150 hover:bg-seal-700"
          >
            返回问答
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-shell flex-1 px-6 py-6">
      <div className="rule-line mb-5 border-b pb-4">
        <h1 className="font-serif text-lg text-ink-900">文档审核</h1>
        <p className="mt-1 text-xs text-ink-500">
          审核通过且索引成功的版本才会进入生效知识库；驳回需填写审核意见留痕。
        </p>
      </div>

      {/* 状态筛选 */}
      <div className="flex flex-wrap items-center gap-2">
        {STATUS_OPTIONS.map((option) => (
          <button
            key={option.value || "all"}
            type="button"
            aria-pressed={status === option.value}
            onClick={() => {
              setPage(1);
              setStatus(option.value);
            }}
            className={cx(
              "rounded border px-3 py-1.5 text-xs transition-colors duration-150",
              status === option.value
                ? "border-seal-300 bg-seal-50 font-medium text-seal-700"
                : "border-ink-200 bg-white text-ink-600 hover:border-ink-300",
            )}
          >
            {option.label}
          </button>
        ))}
      </div>

      {error ? (
        <div className="mt-4">
          <Notice tone="error">{error}</Notice>
        </div>
      ) : null}
      {lastResult ? (
        <div className="mt-4">
          <Notice tone="success">
            {lastResult.decision === "approve"
              ? `审核通过：${lastResult.version_key}，索引 ${lastResult.indexed_chunks ?? 0} 条`
              : `已驳回：${lastResult.version_key}，清理向量 ${lastResult.deleted_vectors ?? 0} 条`}
          </Notice>
        </div>
      ) : null}

      {/* 列表 */}
      {isLoading ? (
        <p className="mt-6 py-10 text-center text-sm text-ink-500">加载中…</p>
      ) : list && list.items.length > 0 ? (
        <div className="mt-4 space-y-3">
          {list.items.map((item) => (
            <ReviewRow
              key={item.version_key}
              item={item}
              reviewing={reviewing}
              reviewNote={reviewNote}
              reviewNoteError={reviewNoteError}
              submitting={submitting}
              onStartReview={startReview}
              onReviewNoteChange={(value) => {
                setReviewNote(value);
                setReviewNoteError(null);
              }}
              onSubmitReview={submitReview}
              onCancelReview={() => setReviewing(null)}
              onOpenDetail={() => void openDetail(item.document_id)}
            />
          ))}
        </div>
      ) : (
        <div className="mt-4">
          <Notice tone="info">当前筛选条件下没有文档版本。</Notice>
        </div>
      )}

      {/* 分页 */}
      {list && list.total > 0 ? (
        <div className="mt-5 flex items-center justify-between text-xs text-ink-500">
          <span>
            共 {list.total} 条 · 第 {list.page} / {totalPages} 页
          </span>
          <div className="flex gap-2">
            <Button
              variant="secondary"
              disabled={page <= 1 || isLoading}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="px-3 py-1.5 text-xs"
            >
              上一页
            </Button>
            <Button
              variant="secondary"
              disabled={page >= totalPages || isLoading}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="px-3 py-1.5 text-xs"
            >
              下一页
            </Button>
          </div>
        </div>
      ) : null}

      {/* 详情侧栏 */}
      {detailLoading || detail || detailError ? (
        <DetailPanel
          detail={detail}
          isLoading={detailLoading}
          error={detailError}
          onClose={() => {
            setDetail(null);
            setDetailError(null);
          }}
        />
      ) : null}
    </div>
  );
}

function ReviewRow({
  item,
  reviewing,
  reviewNote,
  reviewNoteError,
  submitting,
  onStartReview,
  onReviewNoteChange,
  onSubmitReview,
  onCancelReview,
  onOpenDetail,
}: {
  item: AdminReviewListItem;
  reviewing: { documentId: number; decision: "approve" | "reject" } | null;
  reviewNote: string;
  reviewNoteError: string | null;
  submitting: boolean;
  onStartReview: (documentId: number, decision: "approve" | "reject") => void;
  onReviewNoteChange: (value: string) => void;
  onSubmitReview: () => void;
  onCancelReview: () => void;
  onOpenDetail: () => void;
}) {
  const statusTag = STATUS_TAGS[item.version_status];
  const isReviewingHere =
    reviewing !== null && reviewing.documentId === item.document_id;

  return (
    <article className="rounded-lg border border-ink-200 bg-white">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3.5">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-medium text-ink-900">
              {item.title ?? "（无标题）"}
            </h3>
            {item.document_type ? (
              <Tag tone="neutral">{item.document_type}</Tag>
            ) : null}
            {item.version_number ? (
              <Tag tone="unknown">{item.version_number}</Tag>
            ) : null}
            <Tag tone={statusTag.tone}>{statusTag.label}</Tag>
          </div>
          <p className="mt-1 text-xs text-ink-500">
            提交于 {formatDateTime(item.created_at)}
            {item.reviewed_at
              ? ` · 审核于 ${formatDateTime(item.reviewed_at)}`
              : ""}
            {item.review_note ? ` · 意见：${item.review_note}` : ""}
          </p>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <Button
            variant="ghost"
            onClick={onOpenDetail}
            className="px-2.5 py-1.5 text-xs"
          >
            详情
          </Button>
          {item.version_status === "pending_review" ? (
            <>
              <Button
                variant="secondary"
                disabled={submitting}
                onClick={() => onStartReview(item.document_id, "approve")}
                className="px-2.5 py-1.5 text-xs"
              >
                通过
              </Button>
              <Button
                variant="primary"
                disabled={submitting}
                onClick={() => onStartReview(item.document_id, "reject")}
                className="px-2.5 py-1.5 text-xs"
              >
                驳回
              </Button>
            </>
          ) : null}
        </div>
      </div>

      {/* 审核意见填写区（仅点开时出现） */}
      {isReviewingHere ? (
        <div className="border-t border-ink-100 px-4 py-3.5">
          <p className="text-xs font-medium text-ink-700">
            {reviewing?.decision === "approve"
              ? "审核通过（意见选填）"
              : "驳回（审核意见必填，留痕）"}
          </p>
          <textarea
            value={reviewNote}
            onChange={(event) => onReviewNoteChange(event.target.value)}
            rows={2}
            maxLength={1024}
            placeholder={
              reviewing?.decision === "approve"
                ? "例如：已核对官方来源与生效日期"
                : "例如：来源页面与库内版本不一致，需重新采集"
            }
            className="mt-2 w-full rounded-md border border-ink-200 bg-white px-3 py-2 text-sm text-ink-900 placeholder:text-ink-400 focus:border-ink-500 focus:outline-none"
          />
          {reviewNoteError ? (
            <p role="alert" className="mt-1.5 text-xs text-seal-600">
              {reviewNoteError}
            </p>
          ) : null}
          <div className="mt-2 flex items-center gap-2">
            <Button
              onClick={onSubmitReview}
              disabled={submitting}
              className="px-3 py-1.5 text-xs"
            >
              {submitting
                ? "提交中…"
                : reviewing?.decision === "approve"
                  ? "确认通过"
                  : "确认驳回"}
            </Button>
            <Button
              variant="ghost"
              onClick={onCancelReview}
              disabled={submitting}
              className="px-3 py-1.5 text-xs"
            >
              取消
            </Button>
          </div>
        </div>
      ) : null}
    </article>
  );
}

function DetailPanel({
  detail,
  isLoading,
  error,
  onClose,
}: {
  detail: AdminDocumentDetailData | null;
  isLoading: boolean;
  error: string | null;
  onClose: () => void;
}) {
  return (
    <aside
      aria-label="审核详情"
      className="fixed inset-y-0 right-0 z-30 flex w-full max-w-md flex-col border-l border-ink-200 bg-white shadow-lg"
    >
      <div className="rule-line flex items-center justify-between border-b px-4 py-3">
        <h2 className="font-serif text-base text-ink-900">审核详情</h2>
        <Button variant="ghost" onClick={onClose} className="px-2 py-1 text-xs">
          关闭
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {isLoading ? (
          <p className="py-10 text-center text-sm text-ink-500">加载详情…</p>
        ) : error ? (
          <Notice tone="error">{error}</Notice>
        ) : detail ? (
          <>
            <h3 className="text-sm font-medium text-ink-900">
              {detail.title ?? "（无标题）"}
            </h3>
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              {detail.document_type ? (
                <Tag tone="neutral">{detail.document_type}</Tag>
              ) : null}
              <Tag tone={STATUS_TAGS[detail.version_status].tone}>
                {STATUS_TAGS[detail.version_status].label}
              </Tag>
            </div>

            <dl className="mt-4 space-y-1.5 border-t border-ink-100 pt-3.5 text-xs">
              <DetailRow term="版本" value={detail.version_key} />
              <DetailRow term="发布机关" value={detail.issuing_authority ?? "未标注"} />
              <DetailRow
                term="公布日期"
                value={detail.promulgation_date ?? "未标注"}
              />
              <DetailRow term="生效日期" value={detail.effective_date ?? "未标注"} />
              <DetailRow
                term="来源 URL"
                value={
                  detail.source_url ? (
                    <a
                      href={detail.source_url}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="break-all text-seal-600 underline-offset-2 hover:underline"
                    >
                      {detail.source_url}
                    </a>
                  ) : (
                    "未标注"
                  )
                }
              />
              <DetailRow term="内容哈希" value={detail.content_hash} />
              <DetailRow term="提交时间" value={formatDateTime(detail.created_at)} />
            </dl>

            <h4 className="mt-5 text-xs font-medium text-ink-700">
              分块正文预览（{detail.chunk_preview.length} 条）
            </h4>
            <div className="mt-2 space-y-2">
              {detail.chunk_preview.map((chunk) => (
                <div
                  key={chunk.chunk_key}
                  className="rounded-md border border-ink-100 bg-ink-50 px-3 py-2.5"
                >
                  <p className="text-xs text-ink-500">
                    {chunk.article_number ? `第 ${chunk.article_number} 条` : "无条号"}
                  </p>
                  <p className="prose-legal mt-1 whitespace-pre-wrap font-serif text-xs leading-relaxed text-ink-800">
                    {chunk.content}
                  </p>
                </div>
              ))}
              {detail.chunk_preview.length === 0 ? (
                <p className="text-xs text-ink-500">该版本暂无分块正文。</p>
              ) : null}
            </div>

            <h4 className="mt-5 text-xs font-medium text-ink-700">审核留痕</h4>
            {detail.reviewed_by ? (
              <dl className="mt-2 space-y-1.5 text-xs">
                <DetailRow term="审核人" value={detail.reviewed_by} />
                <DetailRow
                  term="审核时间"
                  value={formatDateTime(detail.reviewed_at ?? detail.created_at)}
                />
                <DetailRow term="审核意见" value={detail.review_note ?? "（未填写）"} />
              </dl>
            ) : (
              <p className="mt-2 text-xs text-ink-500">尚未审核。</p>
            )}
          </>
        ) : null}
      </div>
    </aside>
  );
}

function DetailRow({ term, value }: { term: string; value: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <dt className="w-16 shrink-0 text-ink-500">{term}</dt>
      <dd className="min-w-0 flex-1 break-all text-ink-700">{value}</dd>
    </div>
  );
}

/** 后端返回无时区的本地时间字符串，直接按本地时间展示 */
function formatDateTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  return date.toLocaleString("zh-CN", { hour12: false });
}
