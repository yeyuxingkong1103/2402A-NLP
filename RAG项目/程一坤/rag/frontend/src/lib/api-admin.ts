/**
 * 管理端接口（docs/接口文档.md 6.3 / 6.4 / 6.5）。
 *
 * 前端只做「展示与提交」两件事：能否审核、审核结果如何流转都由后端决定，
 * 这里不写任何业务判断。字段与 docs/接口文档.md 严格一致，不自创。
 */

import { requestJson } from "@/lib/api-client";
import type {
  AdminDocumentDetailData,
  AdminReviewListData,
  ReviewDecisionPayload,
  ReviewDecisionResultData,
  ReviewStatus,
} from "@/lib/types";

/** 6.3 审核列表；status 不传则返回全部 */
export function listReviewDocuments(params: {
  status?: ReviewStatus;
  page?: number;
  page_size?: number;
}): Promise<AdminReviewListData> {
  const query = new URLSearchParams();
  if (params.status) {
    query.set("status", params.status);
  }
  if (params.page !== undefined) {
    query.set("page", String(params.page));
  }
  if (params.page_size !== undefined) {
    query.set("page_size", String(params.page_size));
  }
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return requestJson<AdminReviewListData>(`/admin/documents${suffix}`, {
    method: "GET",
  });
}

/** 6.5 审核详情（元数据 + 分块预览） */
export function getReviewDocumentDetail(
  documentId: number,
  previewChunks = 5,
): Promise<AdminDocumentDetailData> {
  return requestJson<AdminDocumentDetailData>(
    `/admin/documents/${documentId}/detail?preview_chunks=${previewChunks}`,
    { method: "GET" },
  );
}

/** 6.4 提交审核决定（approve 意见选填；reject 意见必填由界面与后端共同约束） */
export function submitReviewDecision(
  documentId: number,
  payload: ReviewDecisionPayload,
): Promise<ReviewDecisionResultData> {
  return requestJson<ReviewDecisionResultData>(
    `/admin/documents/${documentId}/review`,
    { method: "POST", body: payload },
  );
}
