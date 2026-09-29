/**
 * 后端接口的类型定义。
 *
 * 全部字段以后端实际返回为准（app/auth/schemas.py、app/api/legal_search.py、
 * app/api/chat_stream.py），与 docs/接口文档.md 的差异处已用注释标明原因。
 * 禁止在组件里就地定义接口返回类型——契约只在这一处描述。
 */

/** 通用响应包（docs/接口文档.md 2.2） */
export interface ApiEnvelope<T> {
  code: number;
  message: string;
  /** 失败时后端返回 null；成功时是业务数据 */
  data: T | null;
  request_id: string;
}

/** 认证成功时 data 的内容（后端 register / login 返回） */
export interface AuthTokenData {
  access_token: string;
  token_type: string;
}

/** 验证码、重置密码、注销时 data 的内容 */
export interface StatusData {
  status: string;
  /** 仅开发环境：后端 SMTP 未接入时回显验证码，生产环境不返回 */
  debug_code?: string;
}

/** 文档类型：API 用英文枚举，后端负责映射成中文业务值 */
export type DocumentType =
  | "law"
  | "administrative_regulation"
  | "judicial_interpretation"
  | "case";

/** 法律检索请求体（app/api/legal_search.py LegalSearchRequest） */
export interface LegalSearchPayload {
  query: string;
  knowledge_base_ids?: string[];
  jurisdiction?: string;
  /** YYYY-MM-DD；不传表示不按时间点过滤 */
  as_of_date?: string | null;
  legal_domain?: string | null;
  document_types?: DocumentType[];
  top_k?: number;
  enable_hybrid_search?: boolean;
  enable_rerank?: boolean;
  session_id?: string | null;
}

/** 检索命中的单条法条（app/api/legal_search.py _serialize_article） */
export interface LegalSearchResultItem {
  chunk_id: string;
  document_id: string | null;
  /** 后端可能返回 null（未知文书类型不在映射表内） */
  document_type: DocumentType | null;
  law_name: string;
  /** 条号，后端以阿拉伯数字存储，如 "47" */
  article_number: string | null;
  paragraph_number: string | null;
  item_number: string | null;
  jurisdiction: string | null;
  effective_date: string | null;
  repeal_date: string | null;
  issuing_authority: string | null;
  is_current: boolean | null;
  source: string;
  content: string;
  /** 候选自身来源的召回分 */
  retrieval_score: number;
  /** 命中来源，取值 vector / keyword，可能同时命中 */
  score_sources: string[];
  vector_score: number | null;
  keyword_score: number | null;
  fusion_score: number | null;
  rerank_score: number | null;
}

/** 查询改写详情（app/api/legal_search.py _serialize_result） */
export interface QueryRewriteInfo {
  original_query: string;
  rewritten_query: string;
  changed: boolean;
  reasons: string[];
}

/** 检索链路的实际生效过滤条件 */
export interface LegalSearchFilters {
  knowledge_base_ids: string[];
  jurisdiction: string;
  as_of_date: string | null;
  legal_domain: string | null;
  document_types: DocumentType[];
}

/** 检索统计：两路召回条数、融合条数、重排后条数 */
export interface RetrievalStats {
  vector_recall_count: number;
  keyword_recall_count: number;
  fused_count: number;
  reranked_count: number;
}

/** 法律检索响应 data */
export interface LegalSearchData {
  query: string;
  rewritten_queries: string[];
  rewrite: QueryRewriteInfo;
  results: LegalSearchResultItem[];
  filters: LegalSearchFilters;
  stats: RetrievalStats;
}

/**
 * 问答流式请求体（app/api/chat_stream.py ChatStreamRequest）。
 *
 * 注意：后端实际路径是 /api/v1/chat/stream，
 * 而 docs/接口文档.md 7.2 写的是 /api/v1/chat/completions —— 以代码为准。
 */
export interface ChatStreamOptions {
  top_k?: number;
  enable_query_rewrite?: boolean;
  enable_long_term_memory?: boolean;
  jurisdiction?: string;
  as_of_date?: string | null;
  document_types?: DocumentType[];
}

export interface ChatStreamPayload {
  session_id: string;
  character_id?: string;
  message: string;
  options?: ChatStreamOptions;
}

/** SSE message_start 事件 */
export interface MessageStartEvent {
  message_id: string;
  request_id: string;
}

/** SSE token 事件 */
export interface TokenEvent {
  text: string;
}

/** SSE citation 事件（app/chat/service.py 组装，只有这 5 个字段） */
export interface CitationEvent {
  chunk_id: string;
  law_name: string;
  article_number: string | null;
  paragraph_number: string | null;
  /** 首期采集与索引没有页码字段，后端明确返回 null */
  page: number | null;
}

/** SSE message_end 事件；后端 LLM 客户端未暴露 usage，因此 usage 恒为 null */
export interface MessageEndEvent {
  finish_reason: string;
  usage: { prompt_tokens: number; completion_tokens: number } | null;
  /** 从 message_start 到回答完成的服务端耗时；旧服务或历史数据可能没有。 */
  elapsed_seconds?: number;
}

/** SSE error 事件 */
export interface StreamErrorEvent {
  code: number;
  message: string;
  retryable: boolean;
  request_id?: string;
}

/**
 * SSE replace 事件（真流式护栏方案 A）：
 * token 已实时发出、撤不回；护栏判定整段回答不可信（如零引用）时，
 * 后端发 replace 携带替换全文，前端用其整段替换当前流式回答内容。
 */
export interface ReplaceEvent {
  text: string;
}

/** 首期固定只有一个预设助手（docs/接口文档.md 7 节） */
export const DEFAULT_CHARACTER_ID = "legal-assistant";

/** 会话列表项（app/api/session_routes.py GET /sessions） */
export interface SessionListItem {
  session_id: string;
  character_id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

/** 会话消息（app/api/session_routes.py GET /sessions/{session_id}/messages） */
export interface SessionMessage {
  message_id: string | null;
  role: "user" | "assistant";
  content: string;
  citations: CitationEvent[] | null;
  model: string | null;
  /** 新版 SSE 结束耗时落库后可用；旧消息没有该字段。 */
  elapsed_seconds?: number | null;
  created_at: string;
}

/** 创建会话响应 data */
export interface CreatedSessionData {
  session_id: string;
  character_id: string;
  title: string;
  created_at: string;
}


/** 首期唯一知识库标识（app/api/legal_search.py KNOWN_KNOWLEDGE_BASE_ID） */
export const DEFAULT_KNOWLEDGE_BASE_ID = "kb_labor_law_001";

/** 首期唯一法律领域（app/api/legal_search.py KNOWN_LEGAL_DOMAIN） */
export const DEFAULT_LEGAL_DOMAIN = "labor_law";

/** 固定免责声明，与后端 guard.py 保持一致，用于界面常驻展示 */
export const RISK_NOTICE =
  "内容仅供法律信息参考，不能替代律师出具的正式法律意见。";

/* ==================== 管理端（docs/接口文档.md 3.8 / 6.3 / 6.4 / 6.5） ==================== */

/** 3.8 当前登录用户信息（不含密码哈希、会话令牌等敏感字段） */
export interface CurrentUserData {
  user_id: string;
  email: string;
  is_admin: boolean;
  long_term_memory_enabled: boolean;
}

/** 6.3 审核状态取值 */
export type ReviewStatus = "pending_review" | "approved" | "rejected";

/** 6.3 审核列表项（title / document_type / version_number 为新增只读字段） */
export interface AdminReviewListItem {
  document_id: number;
  version_key: string;
  version_status: ReviewStatus;
  title: string | null;
  document_type: string | null;
  version_number: string | null;
  created_at: string;
  reviewed_by: string | null;
  reviewed_at: string | null;
  review_note: string | null;
  content_hash: string;
}

/** 6.3 列表响应 data */
export interface AdminReviewListData {
  items: AdminReviewListItem[];
  total: number;
  page: number;
  page_size: number;
}

/** 6.5 审核详情响应 data（版本定位：待审优先，否则最新） */
export interface AdminDocumentDetailData {
  document_id: number;
  version_key: string;
  version_status: ReviewStatus;
  title: string | null;
  document_type: string | null;
  issuing_authority: string | null;
  promulgation_date: string | null;
  effective_date: string | null;
  source_url: string | null;
  content_hash: string;
  created_at: string;
  reviewed_by: string | null;
  reviewed_at: string | null;
  review_note: string | null;
  chunk_preview: Array<{
    chunk_key: string;
    article_number: string | null;
    content: string;
  }>;
}

/** 6.4 审核决定请求体 */
export interface ReviewDecisionPayload {
  decision: "approve" | "reject";
  review_note: string | null;
}

/** 6.4 审核决定响应 data：approve 带 indexed_chunks / index_verified，reject 带 deleted_vectors */
export interface ReviewDecisionResultData {
  decision: "approve" | "reject";
  version_key: string;
  version_status: ReviewStatus;
  reviewed_at: string;
  indexed_chunks?: number;
  index_verified?: boolean;
  deleted_vectors?: number;
}
