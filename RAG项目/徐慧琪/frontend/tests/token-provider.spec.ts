// I1 接线判据（Task 6 首审 Important）：useAuth 模块加载期把 session.token 注入
// request()。删掉那一行时，律师侧所有鉴权请求会静默不带 Bearer——后端 404 →
// auth-expired → 用户被登出跳登录，而原有 117 条全绿。本文件把这条接线钉成两向：
// 律师侧端点必须带 Authorization，公众侧端点必须不带（AC-19 的另一半）。
// 只替 login（它不是判据对象，替身避免真连网），qa/publicQa/request 全走真实实现；
// 网络只 stub fetch——「谁把 header 组出来」留在生产代码里判。
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { login, logout, session } from '@/core/useAuth'
import { qa } from '@/core/api/lawyer'
import { publicQa } from '@/core/api/public'
import { API_PREFIX } from '@/core/config'

// importOriginal 部分替身：保住 qa 等真实导出，只把 login 换成不发网络的替身
vi.mock('@/core/api/lawyer', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/core/api/lawyer')>()
  return {
    ...actual,
    login: vi.fn().mockResolvedValue({
      access_token: 'tok-接线', token_type: 'bearer', role: 'lawyer', team_id: 'team-a',
    }),
  }
})

// 每次请求给新的 Response：体只能读一次，复用同一实例会让第二次请求读到已消费
// 的体（Body is unusable），那是夹具伪影、不是被测行为
function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200, headers: { 'Content-Type': 'application/json' },
  })
}

const fetchMock = vi.fn()

function headersOf(callIndex: number): Record<string, string> {
  return (fetchMock.mock.calls[callIndex][1] as RequestInit).headers as Record<string, string>
}

beforeEach(() => {
  localStorage.clear()
  logout()
  fetchMock.mockReset()
  fetchMock.mockImplementation(async () => jsonResponse({}))
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllGlobals()
  localStorage.clear()
  logout()
})

describe('useAuth → request 的 token 注入（I1 接线）', () => {
  it('登录后 qa() 带 Bearer；publicQa() 即使已登录也不带（两向）', async () => {
    await login('u', 'p')
    expect(session.token).toBe('tok-接线')
    await qa('问题')
    await publicQa('问题')
    expect(fetchMock).toHaveBeenCalledTimes(2)
    // 律师侧端点路径与请求头：Authorization 来自 useAuth 模块加载期的注入
    expect(fetchMock.mock.calls[0][0]).toBe(`${API_PREFIX}/qa`)
    expect(headersOf(0)['Authorization']).toBe('Bearer tok-接线')
    // 公众侧端点不带 Authorization：request.ts 只按 auth 选项取 token，公众端点
    // 不设 auth——接线正确时这里必须是 undefined；反向断言防「一律加头」的写错
    expect(fetchMock.mock.calls[1][0]).toBe(`${API_PREFIX}/public/qa`)
    expect(headersOf(1)['Authorization']).toBeUndefined()
  })
})
