// fetch 封装：错误归型 / 超时 / 取消 / token 注入。页面**不直接 fetch**——
// 网络与错误语义全收在这一个文件，页面只按 ApiError.kind 分支（设计 §三 定调 3）。
//
// 为什么归型要收成一处：后端把「未认证」「限流」「故障」「未实现」编成不同的
// HTTP 状态 + 错误体（④a 契约），而在页面里各写各的 if（status===404 → 跳登录…）
// 时，漏一处的表现是那个页面把「会话失效」显示成「没找到」——没有红灯，只有
// 用户看到的一句错话。kind 是这条语义链上唯一的判据。
import { API_PREFIX, DEFAULT_TIMEOUT_MS } from '@/core/config'

export type ApiErrorKind = 'api' | 'network' | 'timeout' | 'aborted' | 'auth-expired'

export interface ApiErrorInit {
  status?: number
  code?: string
  requestId?: string
  retryAfter?: number | null
}

export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status?: number
  readonly code?: string
  readonly requestId?: string
  readonly retryAfter: number | null

  constructor(kind: ApiErrorKind, message: string, init: ApiErrorInit = {}) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
    this.status = init.status
    this.code = init.code
    this.requestId = init.requestId
    this.retryAfter = init.retryAfter ?? null
  }
}

// token 由**注入**而不是 import：律师侧 token 住在 useAuth（localStorage），
// 而 request.ts 也被公众入口引用——在这里 import useAuth 会把「token 读写」
// 带进公众包，与「公众侧零本地存储」的判据正面冲突。回调没有存储依赖，
// 谁关心 token 谁在启动时把它接上（律师入口 main.ts 接，公众入口不接）。
let tokenProvider: () => string | null = () => null

export function setTokenProvider(fn: () => string | null): void {
  tokenProvider = fn
}

export interface RequestOptions {
  method?: string
  body?: unknown
  timeoutMs?: number
  auth?: boolean
  on404?: 'not-found' | 'auth-expired'
  signal?: AbortSignal
}

// 后端错误体的对外形状（schemas.py 的 errors.PUBLIC_ERRORS 一族；这里只读这两层，
// 多一个字段不认识也不影响归型）
interface ErrorBody {
  request_id?: string
  error?: { code?: string; message?: string }
}

/** 解析错误体；不是 JSON（反代 HTML 页、截断的体…）时给 null——绝不把 HTML 透给用户。 */
async function parseErrorBody(res: Response) {
  try {
    const body = (await res.json()) as ErrorBody | null
    return {
      code: typeof body?.error?.code === 'string' ? body.error.code : undefined,
      message: typeof body?.error?.message === 'string' ? body.error.message : undefined,
      requestId: typeof body?.request_id === 'string' ? body.request_id : undefined,
    }
  } catch {
    return null
  }
}

/** `Retry-After` 的整数秒；缺头或不是纯整数（HTTP-date 等劣构值）一律 null。 */
function parseRetryAfter(raw: string | null): number | null {
  if (raw === null) return null
  const matched = /^\d+$/.exec(raw.trim())
  return matched ? Number(matched[0]) : null
}

export async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const method = opts.method ?? 'GET'
  const controller = new AbortController()
  // 超时与外部取消都会中止 fetch，但归型不同：用**标志**记住是不是自己按的停止。
  // 事后看 controller.signal.aborted 分不出两者（外部取消到达时它也是 true），
  // 而「超时」与「用户取消」在界面上的处理完全相反（提示重试 / 静默回填输入框）。
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, opts.timeoutMs ?? DEFAULT_TIMEOUT_MS)

  const forwardAbort = () => controller.abort()
  if (opts.signal) {
    if (opts.signal.aborted) {
      // 已中止的信号不必再发出请求：各运行时对「带已中止 signal 的 fetch」行为
      // 不一（立刻拒绝或静默挂起），这里直接给结论，行为不随环境漂
      clearTimeout(timer)
      throw new ApiError('aborted', '已取消')
    }
    opts.signal.addEventListener('abort', forwardAbort)
  }

  const headers: Record<string, string> = {}
  // 无请求体就不带 Content-Type（GET 带它会显得像有体；后端也不看它）
  if (opts.body !== undefined) headers['Content-Type'] = 'application/json'
  if (opts.auth) {
    const token = tokenProvider()
    // 无 token 也照发：让后端出 404（未认证）→ on404 归型，而不是在客户端
    // 提前拒绝——「谁有权限」的口径在后端一处，前端不写第二份
    if (token) headers['Authorization'] = `Bearer ${token}`
  }

  let res: Response
  try {
    res = await fetch(API_PREFIX + path, {
      method,
      headers,
      body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
      signal: controller.signal,
    })
  } catch {
    if (timedOut) throw new ApiError('timeout', '请求超时，请稍后重试')
    if (opts.signal?.aborted) throw new ApiError('aborted', '已取消')
    throw new ApiError('network', '网络连接失败，请检查网络后重试')
  } finally {
    clearTimeout(timer)
    opts.signal?.removeEventListener('abort', forwardAbort)
  }

  if (!res.ok) {
    const body = await parseErrorBody(res)
    // 律师侧专用路由上的 404 = 会话失效（④a：未认证访问返回 404 而非 401，
    // 不暴露路径存在性）。后端的 404 文案是「未找到」形态，直接显示就是那句
    // 误导；故这一类用固定文案，其余错误一律透传后端 message（不编造）。
    const authExpired = res.status === 404 && opts.on404 === 'auth-expired'
    throw new ApiError(
      authExpired ? 'auth-expired' : 'api',
      authExpired ? '登录状态已失效，请重新登录'
        : body?.message ?? `请求失败（HTTP ${res.status}）`,
      { status: res.status, code: body?.code, requestId: body?.requestId,
        retryAfter: parseRetryAfter(res.headers.get('Retry-After')) },
    )
  }

  // 204 与空体没有可解析的 JSON：按「无内容」处理，而不是报「解析失败」——
  // 「没有内容」不是错误（后端今天不返回它们，但代理层与将来的 DELETE 会）
  if (res.status === 204) return undefined as T
  let text: string
  try {
    text = await res.text()
  } catch {
    // fetch 已 resolve、读体中途失败：现实来源是用户取消（大响应读到一半）
    if (opts.signal?.aborted) throw new ApiError('aborted', '已取消')
    throw new ApiError('network', '读取响应失败，请重试', { status: res.status })
  }
  if (text.trim() === '') return undefined as T
  try {
    return JSON.parse(text) as T
  } catch {
    // 2xx 却不是 JSON（反代拦截页等）：仍须是 ApiError——页面按 kind 分支，
    // 裸的 SyntaxError 会让它走进「未定义」分支（既不提示也不归一）
    throw new ApiError('api', '响应不是合法 JSON，请稍后重试', { status: res.status })
  }
}
