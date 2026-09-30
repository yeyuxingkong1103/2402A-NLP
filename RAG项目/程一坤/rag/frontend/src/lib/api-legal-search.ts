/**
 * 法律检索接口（app/app/api/legal_search.py）。
 *
 * 该接口不调用大模型，专门用于调试和评估检索链路，
 * 因此前端把它做成独立的「检索调试」页，把所有中间结果暴露出来：
 * 改写后的查询、两路召回条数、融合与重排分数、命中了哪一路。
 */

import { requestJson } from "@/lib/api-client";
import type { LegalSearchData, LegalSearchPayload } from "@/lib/types";

/** 执行一次法律检索 */
export function searchLegalDocuments(
  payload: LegalSearchPayload,
  signal?: AbortSignal,
): Promise<LegalSearchData> {
  return requestJson<LegalSearchData>("/legal/search", {
    body: payload,
    signal,
  });
}

/** 文书类型的中文名，用于界面展示 */
export const DOCUMENT_TYPE_LABELS: Record<string, string> = {
  law: "法律",
  administrative_regulation: "行政法规",
  judicial_interpretation: "司法解释",
  case: "案例材料",
};

/** 命中来源的中文名 */
export const SCORE_SOURCE_LABELS: Record<string, string> = {
  vector: "向量召回",
  keyword: "关键词召回",
  rerank: "重排命中",
};

/** 把条号/款号/项号拼成「第四十七条第二款第一项」式的完整定位 */
export function formatArticleLocation(item: {
  article_number: string | null;
  paragraph_number: string | null;
  item_number: string | null;
}): string {
  const parts: string[] = [];
  if (item.article_number) {
    parts.push(`第${item.article_number}条`);
  }
  if (item.paragraph_number) {
    parts.push(`第${item.paragraph_number}款`);
  }
  if (item.item_number) {
    parts.push(`第${item.item_number}项`);
  }
  return parts.length > 0 ? parts.join("") : "条文定位缺失";
}

/** 生效状态文案；后端 is_current 为 null 表示未知，不能猜成「现行有效」 */
export function describeValidity(item: {
  is_current: boolean | null;
  effective_date: string | null;
  repeal_date: string | null;
}): { label: string; tone: "current" | "expired" | "unknown" } {
  if (item.is_current === true) {
    return { label: "现行有效", tone: "current" };
  }
  if (item.is_current === false) {
    return {
      label: item.repeal_date ? `已失效（${item.repeal_date}）` : "已失效",
      tone: "expired",
    };
  }
  return { label: "生效状态未知", tone: "unknown" };
}
