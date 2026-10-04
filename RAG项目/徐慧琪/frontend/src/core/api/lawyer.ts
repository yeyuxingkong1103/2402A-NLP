// 律师侧专用端点（设计 §三 定调 1：本文件只被律师入口可达）。三条鉴权端点的
// `on404: 'auth-expired'` 就是设计 §五 的那条映射：④a 对未认证访问返回 **404**
// （不暴露路径存在性），只有律师侧专用路由该把它解释成「会话失效」；共享路由
// （法条 / 导航）的 404 是普通的「没找到」，不跳登录。
import { QA_TIMEOUT_MS } from '@/core/config'
import { request } from '@/core/api/request'
import type { components } from '@/core/api-types.gen'

// 下面这四个窄类型与 public.ts 里的那份是**刻意的重复**（Task 2 口径）：结构不变式
// 是单向的——公众入口的模块图里不存在本文件，而律师入口读 public.ts 的 nav/article
// 是允许的（Task 6 审查裁定）；类型都走 import type（编译期擦除），重复不产生运行时
// 字节。字段名同样照抄 schemas.py 的白名单常量，改键时两处同改、由两侧的契约护栏兜住。
export interface Citation {
  article: string | number
  paragraph: string | number | null
  item: string | number | null
  quote: string
}

export interface Source {
  article_no: number
  path: string
  source: string
  rerank_score: number | null
}

export interface SearchBlock {
  article_no: number
  article_no_cn: string | null
  path: string | null
  source: string | null
  rerank_score: number | null
}

// 审计导出行的列：照抄 backend/app/db/audit.py 的 _COLUMNS（ts 由 FastAPI 序列化
// 成 ISO 字符串；user_id / role / team_id 在公众侧为 NULL，故可空）
export interface AuditRow {
  id: number
  ts: string
  request_id: string
  user_id: number | null
  role: string | null
  team_id: string | null
  action: string
  query_text: string | null
  recall_path: string | null
  verify_result: string | null
  status_code: number
  latency_ms: number
}

export type LoginResponse = components['schemas']['LoginResponse']

export type QAAnswer = Omit<components['schemas']['QAAnswer'], 'citations' | 'sources'> & {
  citations: Citation[]
  sources: Source[]
}

export type SearchResponse = Omit<components['schemas']['SearchResponse'], 'blocks'> & {
  blocks: SearchBlock[]
}

export function login(username: string, password: string): Promise<LoginResponse> {
  // 不做本地字段校验：口令两端空格是口令的一部分（schemas.py 的 _Password 不
  // strip），前端替用户 trim 一下就是替后端改了口令
  return request('/auth/login', { method: 'POST', body: { username, password } })
}

export function qa(question: string, signal?: AbortSignal): Promise<QAAnswer> {
  // 与 publicQa 同一笔账：实测 13~24s/发，必须用 QA_TIMEOUT_MS
  return request('/qa', { method: 'POST', body: { question }, auth: true,
    on404: 'auth-expired', timeoutMs: QA_TIMEOUT_MS, signal })
}

export function search(question: string, topK?: number, signal?: AbortSignal): Promise<SearchResponse> {
  // topK 省略时**不发** top_k 键：后端 SearchRequest 的默认值在路由层取
  // （pipeline.RERANK_OUTPUT_TOPK），发一个 null 会让「没传」与「传了 null」
  // 长得一样——今天两者同义，但那是两份默认值分家的起点。
  // signal 是第三参而不是第二参：topK 的位置被 Task 2 冻结接口钉死（既有调用
  // 与端点判据都按它写），取消能力是终审 I4 的授权增量——可选参数、纯加法，
  // 不传时行为与从前逐字节一致（既有 api-endpoints 判据全绿为证）
  const body = topK === undefined ? { question } : { question, top_k: topK }
  return request('/search', { method: 'POST', body, auth: true, on404: 'auth-expired', signal })
}

export function casesSearch(): Promise<never> {
  // 恒 501（后端接口位）：抛 ApiError，页面按 message 展示 reason。
  // 不发请求体是有意的——后端不解析体（输入形状等案件表落地），带上体等于
  // 替一段「未实现」的功能先猜一个形状出来
  return request<never>('/cases/search', { method: 'POST', auth: true,
    on404: 'auth-expired' })
}

export function exportAudit(): Promise<AuditRow[]> {
  // 不传 start/end/limit：缺省窗口（最近 24h、最多 1000 行）是后端契约的一部分，
  // 前端不写第二份默认值；要更早的账时再按同一模式加参数
  return request('/admin/audit/export', { auth: true, on404: 'auth-expired' })
}
