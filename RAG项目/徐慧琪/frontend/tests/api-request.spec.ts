// 错误归型是前端的「错误语义唯一入口」：页面只看 kind 分支。判据覆盖全部 kind
// 与 auth-expired 的触发条件——后者是 ④a「未认证访问律师侧路径返回 404」逼出来的
// （不是 401！前端若按常规 401 处理，token 过期会被显示成「没找到」）。
// （错误路径的请求写成 `request<never>`：这些用例只走失败分支，never 让 catch 的
// 结果收窄成 ApiError——strict 下不写类型实参时 T 推成 unknown，断言要处处再收窄。
// 「抛的必须是 ApiError」仍由各自的 toBeInstanceOf 在运行时判。）
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, request, setTokenProvider } from '@/core/api/request'

function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status, headers: { 'Content-Type': 'application/json', ...headers },
  })
}

beforeEach(() => setTokenProvider(() => null))
afterEach(() => vi.unstubAllGlobals())

describe('request 的错误归型', () => {
  it.each([
    [400, 'bad_request'], [413, 'payload_too_large'], [500, 'internal'],
    [501, 'not_implemented'], [503, 'service_unavailable'],
  ])('HTTP %i → kind=api 且带 code/message/request_id', async (status, code) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      jsonResponse(status, { request_id: 'rid-1', error: { code, message: '后端原文' } })))
    const err = await request<never>('/x').catch((e) => e as ApiError)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.kind).toBe('api')
    expect(err.status).toBe(status)
    expect(err.code).toBe(code)
    expect(err.message).toBe('后端原文')
    expect(err.requestId).toBe('rid-1')
  })

  it('429 读取 Retry-After（秒）', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      jsonResponse(429, { error: { code: 'rate_limited', message: '慢点' } },
        { 'Retry-After': '30' })))
    const err = await request<never>('/x').catch((e) => e as ApiError)
    expect(err.kind).toBe('api')
    expect(err.retryAfter).toBe(30)
  })

  it('Retry-After 是劣构值（HTTP-date）→ retryAfter=null（不把 NaN 传给倒计时）', async () => {
    // parseRetryAfter 只认纯整数秒。缺这条判据时，一个朴素 Number(raw) 实现会把
    // HTTP-date 解析成 NaN 并混过全部测试——页面拿到 NaN 去算倒计时显示成「NaN 秒后重试」
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      jsonResponse(429, { error: { code: 'rate_limited', message: '慢点' } },
        { 'Retry-After': 'Wed, 21 Oct 2015 07:28:00 GMT' })))
    const err = await request<never>('/x').catch((e) => e as ApiError)
    expect(err.kind).toBe('api')
    expect(err.retryAfter).toBeNull()
  })

  it('404 + on404=auth-expired → kind=auth-expired（律师侧专用路由）', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      jsonResponse(404, { request_id: 'r', error: { code: 'not_found', message: '未找到' } })))
    const err = await request<never>('/x', { auth: true, on404: 'auth-expired' }).catch((e) => e as ApiError)
    expect(err.kind).toBe('auth-expired')
    expect(err.status).toBe(404)
  })

  it('404 + 默认 → 普通 api 错误（共享路由的「未找到」不跳登录）', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      jsonResponse(404, { error: { code: 'not_found', message: '未找到' } })))
    const err = await request<never>('/x').catch((e) => e as ApiError)
    expect(err.kind).toBe('api')
    expect(err.code).toBe('not_found')
  })

  it('fetch 拒绝 → kind=network', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    expect((await request<never>('/x').catch((e) => e as ApiError)).kind).toBe('network')
  })

  it('非 JSON 错误体（反代 HTML 页）→ 通用文案，绝不透传 body 文本', async () => {
    // 反代/网关出错时给的是 HTML 而不是后端的 JSON 错误体。此时必须落到通用文案：
    // 把 `<html>…` 原样当 message 显示既无意义，又会把网关内部信息漏给用户
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '<html><body>502 Bad Gateway</body></html>',
      { status: 502, headers: { 'Content-Type': 'text/html' } })))
    const err = await request<never>('/x').catch((e) => e as ApiError)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.kind).toBe('api')
    expect(err.status).toBe(502)
    expect(err.message).toBe('请求失败（HTTP 502）')
    expect(err.message).not.toContain('<html>')
  })

  it('超时 → kind=timeout（自己触发的 abort 与外部取消要分开）', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn((_u: string, init: RequestInit) => new Promise((_r, rej) => {
      init.signal?.addEventListener('abort', () => rej(new DOMException('a', 'AbortError')))
    })))
    const p = request<never>('/x', { timeoutMs: 100 }).catch((e) => e as ApiError)
    await vi.advanceTimersByTimeAsync(101)
    expect((await p).kind).toBe('timeout')
    vi.useRealTimers()
  })

  it('响应到达后超时定时器被清理（不为每次成功请求留一个挂起的计时器）', async () => {
    // 成功路径靠 finally 里的 clearTimeout 收尾。删掉它其它用例全绿，但每次成功
    // 请求都会在事件循环里留一个定时器直到超时窗口结束（默认 15s），长会话下越积
    // 越多；这里用假计时器把「残留」变成可直接观测的计数
    vi.useFakeTimers()
    try {
      vi.stubGlobal('fetch', vi.fn().mockImplementation(async () => jsonResponse(200, {})))
      await request('/x')
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      // 断言失败时也要把假计时器还回去，避免影响同一文件里后续用例（变异验证依赖这点）
      vi.useRealTimers()
    }
  })

  it('外部取消 → kind=aborted', async () => {
    vi.stubGlobal('fetch', vi.fn((_u: string, init: RequestInit) => new Promise((_r, rej) => {
      init.signal?.addEventListener('abort', () => rej(new DOMException('a', 'AbortError')))
    })))
    const ac = new AbortController()
    const p = request<never>('/x', { signal: ac.signal }).catch((e) => e as ApiError)
    ac.abort()
    expect((await p).kind).toBe('aborted')
  })

  it('auth:true 带 Bearer；auth:false 即使有 token 也不带', async () => {
    // 每次调用给**新的** Response：Response 的体只能读一次，mockResolvedValue 复用
    // 同一个实例会让第二次请求读到已消费的体（TypeError: Body is unusable），
    // 那是夹具伪影、不是被测行为——这里只关心两次请求各带了什么头
    const spy = vi.fn().mockImplementation(async () => jsonResponse(200, {}))
    vi.stubGlobal('fetch', spy)
    setTokenProvider(() => 'tok-1')
    await request('/a', { auth: true })
    await request('/b')
    expect(spy.mock.calls[0][1].headers['Authorization']).toBe('Bearer tok-1')
    expect(spy.mock.calls[1][1].headers['Authorization']).toBeUndefined()
  })
})
