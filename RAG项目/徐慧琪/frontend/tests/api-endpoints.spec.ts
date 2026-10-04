// 端点 × 鉴权 × 404 语义 的表驱动判据（设计 §三 定调 3 的那张表）。没有这条：
// 某个端点把 auth 标志抄漏时，请求照发、测试照绿——直到真机上「未登录的律师侧
// 调用拿到的是普通 404 而不是跳登录」或「公众端点悄悄带上了 token」才显形。
// 每条断言的是**请求的形状**（method / path / body / Authorization），不是响应。
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, setTokenProvider } from '@/core/api/request'
import { article, nav, publicQa } from '@/core/api/public'
import { casesSearch, exportAudit, login, qa, search } from '@/core/api/lawyer'

const TOKEN = 'tok-1'

// 每次调用都给新的 Response：Response 的体只能读一次，复用同一个实例会让第二条
// 请求撞上「Body is unusable」。用真 Response 而不是手搓替身，是为了让
// request.ts 的成功路径（读体 → JSON.parse）也走真实分支
function jsonResponse(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status, headers: { 'Content-Type': 'application/json' },
  })
}

interface Endpoint {
  name: string
  call: () => Promise<unknown>
  method: string
  path: string
  body?: unknown
  auth: boolean
  // 该端点 404 的归型（设计 §五）：律师侧专用路由 = 会话失效；公众/共享路由 = 普通
  // 「未找到」。两向都要钉——漏配 on404 与多配 on404 是同一张表上的两种写反
  notFoundKind: 'api' | 'auth-expired'
}

// 逐列对上计划里的「端点 × 鉴权 × 404 语义」表；body 为 undefined = 不发请求体
const ENDPOINTS: Endpoint[] = [
  { name: 'login', call: () => login('lawyer1', 'pw-1'), method: 'POST',
    path: '/api/v1/auth/login', body: { username: 'lawyer1', password: 'pw-1' },
    auth: false, notFoundKind: 'api' },
  { name: 'publicQa', call: () => publicQa('我可以退租吗'), method: 'POST',
    path: '/api/v1/public/qa', body: { question: '我可以退租吗' },
    auth: false, notFoundKind: 'api' },
  { name: 'article', call: () => article('minfadian', 584), method: 'GET',
    path: '/api/v1/law/minfadian/articles/584', auth: false, notFoundKind: 'api' },
  { name: 'nav', call: () => nav(), method: 'GET', path: '/api/v1/law/nav',
    auth: false, notFoundKind: 'api' },
  { name: 'qa', call: () => qa('我可以退租吗'), method: 'POST',
    path: '/api/v1/qa', body: { question: '我可以退租吗' },
    auth: true, notFoundKind: 'auth-expired' },
  { name: 'search', call: () => search('我可以退租吗', 3), method: 'POST',
    path: '/api/v1/search', body: { question: '我可以退租吗', top_k: 3 },
    auth: true, notFoundKind: 'auth-expired' },
  // casesSearch 无请求体（后端不解析体，见 lawyer.ts 的注释）
  { name: 'casesSearch', call: () => casesSearch(), method: 'POST',
    path: '/api/v1/cases/search', auth: true, notFoundKind: 'auth-expired' },
  { name: 'exportAudit', call: () => exportAudit(), method: 'GET',
    path: '/api/v1/admin/audit/export', auth: true, notFoundKind: 'auth-expired' },
]

let fetchSpy: ReturnType<typeof vi.fn>

beforeEach(() => {
  // 装上 token 再跑：auth=false 的行能一并钉住「即使有 token 也不带 Authorization」
  setTokenProvider(() => TOKEN)
  fetchSpy = vi.fn().mockImplementation(async () => jsonResponse(200, {}))
  vi.stubGlobal('fetch', fetchSpy)
})

afterEach(() => vi.unstubAllGlobals())

describe('端点请求形状', () => {
  it.each(ENDPOINTS)('$name：$method $path（auth=$auth）', async ({ call, method, path, body, auth }) => {
    await call()
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit]
    expect(url).toBe(path)
    expect(init.method).toBe(method)
    const headers = init.headers as Record<string, string>
    expect(headers['Authorization']).toBe(auth ? `Bearer ${TOKEN}` : undefined)
    const sent = init.body === undefined ? undefined : JSON.parse(init.body as string)
    expect(sent).toEqual(body)
  })

  it.each(ENDPOINTS)('$name：404 的归型是 $notFoundKind', async ({ call, notFoundKind }) => {
    // 端点自己的 404 语义（设计 §五 的中央映射）：律师侧专用路由 404 = 会话失效
    // → 跳登录；公众/共享路由 404 = 普通「未找到」→ 不跳登录。没有这条时，
    // 某端点漏配（或多配）on404 全绿——它的页面会把「会话失效」显示成「没找到」，
    // 正是本 spec 头部注释宣告要防的那种静默失效。两向都断言：删掉律师侧的
    // on404 与给公众侧错加 on404，都会在这里红
    fetchSpy.mockImplementation(async () => jsonResponse(404, {
      request_id: 'r', error: { code: 'not_found', message: '未找到' },
    }))
    const err = (await call().catch((e: unknown) => e)) as ApiError
    expect(err).toBeInstanceOf(ApiError)
    expect(err.status).toBe(404)
    expect(err.kind).toBe(notFoundKind)
  })

  it('search 省略 topK 时不发 top_k 键（默认值只有后端一处）', async () => {
    await search('我可以退租吗')
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit]
    expect(JSON.parse(init.body as string)).toEqual({ question: '我可以退租吗' })
  })

  it('search 把第三参 signal 透传到 request：已中止即 aborted 且不发 fetch（终审 I4）', async () => {
    // 页面级的取消判据只钉到「search 收到 signal」；这条钉 API 层不把它丢掉：
    // 已中止的 signal 在 request.ts 里直接归型 aborted、根本不发请求——删掉
    // lawyer.ts 里的 signal 透传后 fetch 会被调用（或归型不是 aborted），必红
    const c = new AbortController()
    c.abort()
    const err = await search('我可以退租吗', undefined, c.signal).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).kind).toBe('aborted')
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it('casesSearch：已认证仍 501 → 抛带 reason 的 ApiError（接口位联调）', async () => {
    fetchSpy.mockImplementation(async () => jsonResponse(501, {
      request_id: 'rid-501',
      error: { code: 'not_implemented', message: '历史案件检索尚未实现（FR-5.3）' },
    }))
    const err = await casesSearch().catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.kind).toBe('api')
    expect(err.status).toBe(501)
    expect(err.code).toBe('not_implemented')
    expect(err.message).toBe('历史案件检索尚未实现（FR-5.3）')
  })
})
