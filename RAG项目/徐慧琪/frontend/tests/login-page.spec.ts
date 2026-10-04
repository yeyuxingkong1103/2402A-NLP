// LoginPage 此前零判据（终审 m8/I3）：登录成功回跳、后端 message 透传、429
// 倒计时内禁提交与解除、以及「倒计时内点重试不得死锁」这条承重面全在这里钉。
// 状态机与公众 AskPage / 律师 WorkbenchPage 同构——三页各有一份实现，各自有判据。
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createMemoryHistory } from 'vue-router'
import LoginPage from '@/lawyer/pages/LoginPage.vue'
import { createLawyerRouter } from '@/lawyer/router'
import { ApiError } from '@/core/api/request'
import { login as apiLogin } from '@/core/api/lawyer'
import { logout } from '@/core/useAuth'

// 工厂式 mock：router.ts 会拉起全部六个页面，五个端点一并给出，避免未使用的
// 页面 import 到 undefined（vitest 对缺失导出直接报错）
vi.mock('@/core/api/lawyer', () => ({
  login: vi.fn(), qa: vi.fn(), search: vi.fn(), casesSearch: vi.fn(), exportAudit: vi.fn(),
}))
vi.mock('@/core/api/public', () => ({ publicQa: vi.fn(), article: vi.fn(), nav: vi.fn() }))

const loginMock = vi.mocked(apiLogin)

function makeRouter() {
  return createLawyerRouter(createMemoryHistory())
}

const OK = { access_token: 't', token_type: 'bearer', role: 'lawyer', team_id: 'team-a' }

// jsdom 没有 navigator.clipboard：复制判据装替身（终审 I2），afterEach 摘掉
function stubClipboard() {
  const writeText = vi.fn().mockResolvedValue(undefined)
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
  return writeText
}

async function fillAndSubmit(w: ReturnType<typeof mount>, user = 'u', pw = 'p') {
  await w.get('[data-test=username]').setValue(user)
  await w.get('[data-test=password]').setValue(pw)
  await w.get('[data-test=submit]').trigger('click')
}

beforeEach(() => {
  localStorage.clear()
  logout()
  loginMock.mockReset()
})

// 429 用例中途换假表；不还原会把后续文件里的真实等待全冻住
afterEach(() => {
  vi.useRealTimers()
  Reflect.deleteProperty(navigator, 'clipboard')
})

describe('LoginPage（律师登录）', () => {
  it('登录成功 → 跳守卫记下的 returnTo（直接访问 /cases 被拦的场景）', async () => {
    loginMock.mockResolvedValue(OK)
    const router = makeRouter()
    await router.push('/login?returnTo=/cases')
    const w = mount(LoginPage, { global: { plugins: [router] } })
    await fillAndSubmit(w, 'lawyer1', 'pw-1')
    await flushPromises()
    expect(loginMock).toHaveBeenCalledWith('lawyer1', 'pw-1')
    expect(router.currentRoute.value.path).toBe('/cases')
  })

  it('登录成功且无 returnTo → 落工作台（负向：不跳一个有 query 的假目标）', async () => {
    loginMock.mockResolvedValue(OK)
    const router = makeRouter()
    await router.push('/login')
    const w = mount(LoginPage, { global: { plugins: [router] } })
    await fillAndSubmit(w)
    await flushPromises()
    expect(router.currentRoute.value.path).toBe('/')
  })

  it('后端错误 → ErrorPanel 透传后端 message，不跳转', async () => {
    loginMock.mockRejectedValue(
      new ApiError('api', '用户名或密码错误', { status: 400, code: 'bad_request' }))
    const router = makeRouter()
    await router.push('/login')
    const w = mount(LoginPage, { global: { plugins: [router] } })
    await fillAndSubmit(w)
    await flushPromises()
    // 文案不编造：显示的就是后端那句，而不是前端自造的「登录失败」
    expect(w.get('[data-test=error]').text()).toContain('用户名或密码错误')
    expect(router.currentRoute.value.path).toBe('/login')
  })

  it('错误面板复制 → 剪贴板收到 request_id（终审 I2，去掉 @copied 必红）', async () => {
    const writeText = stubClipboard()
    loginMock.mockRejectedValue(new ApiError('api', '服务暂时不可用',
      { status: 503, code: 'service_unavailable', requestId: 'rid-login' }))
    const w = mount(LoginPage, { global: { plugins: [makeRouter()] } })
    await fillAndSubmit(w)
    await flushPromises()
    await w.get('[data-test=copy-id]').trigger('click')
    expect(writeText).toHaveBeenCalledWith('rid-login')
  })

  it('429（retryAfter=30）→ 提交按钮禁用且显示倒计时（删掉禁用接线必红）', async () => {
    vi.useFakeTimers()
    loginMock.mockRejectedValue(
      new ApiError('api', '请求过于频繁', { status: 429, code: 'rate_limited', retryAfter: 30 }))
    const w = mount(LoginPage, { global: { plugins: [makeRouter()] } })
    await fillAndSubmit(w)
    // 假表下不能用 flushPromises（依赖真 setTimeout）：推进 0ms 让 rejection 落进 catch
    await vi.advanceTimersByTimeAsync(0)
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeDefined()
    expect(w.get('[data-test=countdown]').text()).toContain('30 秒后可重试')
    w.unmount()
  })

  it('倒计时走完 → 解除禁用（恢复可提交）', async () => {
    vi.useFakeTimers()
    loginMock.mockRejectedValue(
      new ApiError('api', '请求过于频繁', { status: 429, code: 'rate_limited', retryAfter: 2 }))
    const w = mount(LoginPage, { global: { plugins: [makeRouter()] } })
    await fillAndSubmit(w)
    await vi.advanceTimersByTimeAsync(0)
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeDefined()
    await vi.advanceTimersByTimeAsync(2000)
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeUndefined()
    w.unmount()
  })

  it('429 倒计时内点重试成功 → 立即解锁，不等随卸载消失的倒计时（起点复位，不得死锁）', async () => {
    // 倒计时中点重试会卸载 ErrorPanel，countdown-end 随组件一起消失不再发；
    // 若 submit() 不在起点复位 rateLimited，这次重试成功之后登录按钮会永久
    // 置灰（只能刷新页面）。与 AskPage/WorkbenchPage 的 I1 用例同构
    vi.useFakeTimers()
    loginMock
      .mockRejectedValueOnce(new ApiError('api', '请求过于频繁', { status: 429, code: 'rate_limited', retryAfter: 30 }))
      .mockResolvedValueOnce(OK)
    const router = makeRouter()
    await router.push('/login')
    const w = mount(LoginPage, { global: { plugins: [router] } })
    await fillAndSubmit(w)
    await vi.advanceTimersByTimeAsync(0)
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeDefined()
    await w.get('[data-test=retry]').trigger('click')
    await vi.advanceTimersByTimeAsync(0)
    expect(router.currentRoute.value.path).toBe('/') // 重试成功：回跳工作台
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeUndefined()
    // 再推进 45 秒（远超 Retry-After）仍可提交：解锁只能来自本次提交的复位，
    // 而不是某个稍后才可能到达的 countdown-end（那条路已被卸载切断）
    await vi.advanceTimersByTimeAsync(45000)
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeUndefined()
    w.unmount()
  })
})
