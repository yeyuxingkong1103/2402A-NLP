// CasesPage（501 联调位）与 AuditPage（partner 门 + 导出下载）。API 全 mock：
// 页面判的是「后端说什么就显示什么」与「谁能看到入口」，真实归型另在
// api-endpoints spec 判。
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createMemoryHistory } from 'vue-router'
import AuditPage from '@/lawyer/pages/AuditPage.vue'
import CasesPage from '@/lawyer/pages/CasesPage.vue'
import { createLawyerRouter } from '@/lawyer/router'
import { ApiError } from '@/core/api/request'
import { logout, session } from '@/core/useAuth'
import { casesSearch, exportAudit } from '@/core/api/lawyer'
import type { AuditRow } from '@/core/api/lawyer'

vi.mock('@/core/api/lawyer', () => ({
  login: vi.fn(), qa: vi.fn(), search: vi.fn(), casesSearch: vi.fn(), exportAudit: vi.fn(),
}))

const casesMock = vi.mocked(casesSearch)
const exportMock = vi.mocked(exportAudit)

// jsdom 没有 navigator.clipboard：复制判据装替身（终审 I2），afterEach 摘掉
function stubClipboard() {
  const writeText = vi.fn().mockResolvedValue(undefined)
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
  return writeText
}

function makeRouter() {
  return createLawyerRouter(createMemoryHistory())
}

const ROWS: AuditRow[] = [{
  id: 1, ts: '2026-10-03T00:00:00Z', request_id: 'r-1', user_id: 1, role: 'partner',
  team_id: 'team-a', action: 'qa', query_text: '违约怎么办', recall_path: null,
  verify_result: null, status_code: 200, latency_ms: 120,
}]

// jsdom 没有 URL.createObjectURL（Blob 下载的唯一把手）：装替身并记下
// anchor.click 时的 download 属性——文件名是契约（验收按它找文件）
const createObjectURL = vi.fn((_blob: Blob) => 'blob:test')
const revokeObjectURL = vi.fn()
let downloadName = ''

beforeEach(() => {
  localStorage.clear()
  logout()
  casesMock.mockReset()
  exportMock.mockReset()
  createObjectURL.mockClear()
  downloadName = ''
  Object.defineProperty(URL, 'createObjectURL', { value: createObjectURL, configurable: true })
  Object.defineProperty(URL, 'revokeObjectURL', { value: revokeObjectURL, configurable: true })
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    downloadName = this.download
  })
})

afterEach(() => {
  vi.restoreAllMocks()
  Reflect.deleteProperty(navigator, 'clipboard')
  Reflect.deleteProperty(URL, 'createObjectURL')
  Reflect.deleteProperty(URL, 'revokeObjectURL')
})

describe('CasesPage（历史案件）', () => {
  it('501 → 展示后端 message 与错误代码，不编造「暂无数据」', async () => {
    casesMock.mockRejectedValue(new ApiError('api', '历史案件检索尚未实现（FR-5.3）',
      { status: 501, code: 'not_implemented', requestId: 'rid-501' }))
    const w = mount(CasesPage, { global: { plugins: [makeRouter()] } })
    await flushPromises()
    expect(w.get('[data-test=error]').text()).toContain('历史案件检索尚未实现（FR-5.3）')
    expect(w.get('[data-test=error]').text()).toContain('not_implemented')
  })

  it('错误面板的重试重发请求（接口位可反复联调）', async () => {
    casesMock.mockRejectedValue(new ApiError('api', '历史案件检索尚未实现（FR-5.3）', { status: 501 }))
    const w = mount(CasesPage, { global: { plugins: [makeRouter()] } })
    await flushPromises()
    expect(casesMock).toHaveBeenCalledTimes(1)
    await w.get('[data-test=retry]').trigger('click')
    await flushPromises()
    expect(casesMock).toHaveBeenCalledTimes(2)
  })

  it('501 错误面板复制 → 剪贴板收到 request_id（终审 I2，去掉 @copied 必红）', async () => {
    const writeText = stubClipboard()
    casesMock.mockRejectedValue(new ApiError('api', '历史案件检索尚未实现（FR-5.3）',
      { status: 501, code: 'not_implemented', requestId: 'rid-501' }))
    const w = mount(CasesPage, { global: { plugins: [makeRouter()] } })
    await flushPromises()
    await w.get('[data-test=copy-id]').trigger('click')
    expect(writeText).toHaveBeenCalledWith('rid-501')
  })
})

describe('AuditPage（审计导出）', () => {
  it("role='lawyer' → 不渲染导出入口（只看到权限说明）", async () => {
    session.role = 'lawyer'
    const w = mount(AuditPage, { global: { plugins: [makeRouter()] } })
    expect(w.find('[data-test=export]').exists()).toBe(false)
    expect(w.find('[data-test=denied]').exists()).toBe(true)
  })

  it("role='partner' → 渲染导出入口", async () => {
    session.role = 'partner'
    const w = mount(AuditPage, { global: { plugins: [makeRouter()] } })
    expect(w.find('[data-test=export]').exists()).toBe(true)
    expect(w.find('[data-test=denied]').exists()).toBe(false)
  })

  it('导出 → Blob 下载被触发，文件名 audit-export.json', async () => {
    session.role = 'partner'
    exportMock.mockResolvedValue(ROWS)
    const w = mount(AuditPage, { global: { plugins: [makeRouter()] } })
    await w.get('[data-test=export]').trigger('click')
    await flushPromises()
    expect(exportMock).toHaveBeenCalledTimes(1)
    expect(createObjectURL).toHaveBeenCalledTimes(1)
    expect(createObjectURL.mock.calls[0][0]).toBeInstanceOf(Blob)
    expect(downloadName).toBe('audit-export.json')
    expect(w.get('[data-test=done]').text()).toContain('已导出 1 条')
  })
})

// 会话失效接线（Task 6 首审 m3：Workbench 一侧有刀承重，这两页没有）——
// 404 归型 auth-expired 时必须清 token 跳登录；去掉页面里的 handleAuthExpired
// 接线后既有 5 条仍绿，但用户会看到普通错误面板且 token 还留着
describe('CasesPage / AuditPage 的 handleAuthExpired 接线', () => {
  it('CasesPage：auth-expired → 清 token 跳登录，不弹错误面板', async () => {
    session.token = 'tok-c'
    localStorage.setItem('fl_lawyer_session', JSON.stringify({
      token: 'tok-c', role: 'lawyer', teamId: 'team-c', username: 'u-c',
    }))
    casesMock.mockRejectedValue(new ApiError('auth-expired', '登录状态已失效，请重新登录', { status: 404 }))
    const router = makeRouter()
    await router.push('/cases')
    const w = mount(CasesPage, { global: { plugins: [router] } })
    await flushPromises()
    expect(session.token).toBeNull()
    expect(localStorage.getItem('fl_lawyer_session')).toBeNull()
    expect(router.currentRoute.value.path).toBe('/login')
    expect(w.find('[data-test=error]').exists()).toBe(false)
  })

  it('AuditPage：导出遇 auth-expired → 清 token 跳登录', async () => {
    session.token = 'tok-a'
    session.role = 'partner'
    localStorage.setItem('fl_lawyer_session', JSON.stringify({
      token: 'tok-a', role: 'partner', teamId: 'team-a', username: 'u-a',
    }))
    exportMock.mockRejectedValue(new ApiError('auth-expired', '登录状态已失效，请重新登录', { status: 404 }))
    const router = makeRouter()
    await router.push('/audit')
    const w = mount(AuditPage, { global: { plugins: [router] } })
    await w.get('[data-test=export]').trigger('click')
    await flushPromises()
    expect(session.token).toBeNull()
    expect(localStorage.getItem('fl_lawyer_session')).toBeNull()
    expect(router.currentRoute.value.path).toBe('/login')
  })
})

// 角色门契约（Task 6 首审 m4）：后端 Role 有 lawyer/partner/assistant 三值，
// 既有用例只钉了前两个——门写成 `role !== 'lawyer'` 时 assistant 会看到入口，
// 点击后被后端 require_roles(PARTNER) 拒绝（404 → 前端登出），旧套件无红灯
describe('AuditPage 角色门（契约第三角色）', () => {
  it("role='assistant' → 不渲染导出入口（门只认 partner）", async () => {
    session.role = 'assistant'
    const w = mount(AuditPage, { global: { plugins: [makeRouter()] } })
    expect(w.find('[data-test=export]').exists()).toBe(false)
    expect(w.find('[data-test=denied]').exists()).toBe(true)
  })
})
