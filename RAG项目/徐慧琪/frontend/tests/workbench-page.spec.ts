// WorkbenchPage 是律师侧主页：问答/检索两模式、复制落点、会话失效处理与 429
// 状态机（与公众 AskPage 同形，裁决 A）。API 全 mock——页面测试判的是**接线**，
// 真实的网络与错误归型语义在 api-request / api-endpoints spec 里另判。
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createMemoryHistory } from 'vue-router'
import WorkbenchPage from '@/lawyer/pages/WorkbenchPage.vue'
import { createLawyerRouter } from '@/lawyer/router'
import { ApiError } from '@/core/api/request'
import { login, logout, session } from '@/core/useAuth'
import { login as apiLogin, qa, search } from '@/core/api/lawyer'
import type { QAAnswer, SearchResponse } from '@/core/api/lawyer'

// 工厂式 mock：本 spec 的模块图里六个页面都可能被 router.ts 拉起，五个端点
// 一并给出，避免未使用的页面 import 到 undefined（vitest 对缺失导出会直接报错）
vi.mock('@/core/api/lawyer', () => ({
  login: vi.fn(), qa: vi.fn(), search: vi.fn(), casesSearch: vi.fn(), exportAudit: vi.fn(),
}))
// 共享端点（nav/article，来自 core/api/public）的判据已拆去
// lawyer-nav-article-page.spec.ts；router.ts 的模块图仍会拉起律师 Nav/Article
// 两页，mock 必须给全导出，否则这两页在 import 期就拿到 undefined
vi.mock('@/core/api/public', () => ({ publicQa: vi.fn(), article: vi.fn(), nav: vi.fn() }))

const qaMock = vi.mocked(qa)
const searchMock = vi.mocked(search)

function answer(over: Partial<QAAnswer> = {}): QAAnswer {
  return {
    request_id: 'r-1',
    status: 'ok',
    answer: '依据民法典，可以退租。',
    citations: [{ article: '584', paragraph: null, item: null,
      quote: '当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担违约责任。' }],
    sources: [],
    disclaimer: '本回答仅为一般性法律信息，不构成正式法律意见，具体案件请咨询律师。',
    failures: [],
    ...over,
  }
}

function searchFixture(over: Partial<SearchResponse> = {}): SearchResponse {
  return {
    request_id: 'r-s',
    question: '违约金过高怎么办',
    blocks: [{ article_no: 584, article_no_cn: '第五百八十四条',
      path: '第三编 合同 > 第一分编 通则 > 第四章 合同的履行', source: '民法典', rerank_score: 0.87 }],
    exact_nos: [584],
    ...over,
  }
}

function makeRouter() {
  return createLawyerRouter(createMemoryHistory())
}

async function ask(w: ReturnType<typeof mount>, text: string) {
  await w.get('[data-test=input]').setValue(text)
  await w.get('[data-test=submit]').trigger('click')
  await flushPromises()
}

// jsdom 没有 navigator.clipboard：复制判据要自己装替身（裁决 C）。
// configurable 让 afterEach 能整体摘掉，不把替身漏给别的用例
function stubClipboard() {
  const writeText = vi.fn().mockResolvedValue(undefined)
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
  return writeText
}

beforeEach(() => {
  localStorage.clear()
  logout()
  qaMock.mockReset()
  searchMock.mockReset()
})

afterEach(() => {
  vi.useRealTimers()
  Reflect.deleteProperty(navigator, 'clipboard')
})

describe('WorkbenchPage（律师工作台）', () => {
  it('问答模式：提交 → qa() 被调、AnswerCard 出现（解读 + 原文分区）', async () => {
    qaMock.mockResolvedValue(answer())
    const w = mount(WorkbenchPage, { global: { plugins: [makeRouter()] } })
    await ask(w, '  合同违约怎么办？  ')
    // QuestionBox 的 trim 协议在页面这一层仍然成立（mock 收到 trim 后问句）
    expect(qaMock).toHaveBeenCalledTimes(1)
    expect(qaMock.mock.calls[0][0]).toBe('合同违约怎么办？')
    const card = w.get('[data-test=result]')
    expect(card.text()).toContain('AI 解读，仅供参考')
    expect(card.text()).toContain('法条原文')
    expect(card.text()).toContain('当事人一方不履行合同义务')
  })

  it('点复制 → 剪贴板收到 buildCopyText 的字面量结果（解读 + 空行 + 依据）', async () => {
    const writeText = stubClipboard()
    qaMock.mockResolvedValue(answer())
    const w = mount(WorkbenchPage, { global: { plugins: [makeRouter()] } })
    await ask(w, '合同违约怎么办？')
    await w.get('.answer-card__copy').trigger('click')
    // 字面量锚（FR-7.4）：复制文本改一个字符这里就红——不调用 buildCopyText
    // 来生成期望值，否则格式函数怎么改都全绿
    expect(writeText).toHaveBeenCalledWith(
      '依据民法典，可以退租。\n\n《中华人民共和国民法典》第584条：“当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担违约责任。”')
  })

  it('切到检索模式：search() 被调，blocks 与 exact_nos 标注渲染，qa 不参与', async () => {
    searchMock.mockResolvedValue(searchFixture())
    const w = mount(WorkbenchPage, { global: { plugins: [makeRouter()] } })
    await w.get('[data-test=mode-search]').trigger('click')
    await ask(w, '违约金过高怎么办')
    expect(qaMock).not.toHaveBeenCalled()
    expect(searchMock).toHaveBeenCalledTimes(1)
    expect(searchMock.mock.calls[0][0]).toBe('违约金过高怎么办')
    const block = w.get('[data-test=block]')
    expect(block.text()).toContain('第五百八十四条')
    expect(block.text()).toContain('第三编 合同 > 第一分编 通则 > 第四章 合同的履行')
    expect(block.text()).toContain('0.87')
    expect(w.get('[data-test=exact-no]').text()).toBe('精确置顶：第 584 条')
  })

  it('检索模式取消：search 收到已中止的 signal，不产生结果也不报错（终审 I4）', async () => {
    // mock 复刻 request.ts 的取消归型：signal 中止 → ApiError(aborted)。
    // 修复前 search 不接受 signal，检索中的「取消」是无效果动作；现在页面把
    // controller.signal 透传第三参（topK 仍不传，默认值只有后端一处）
    searchMock.mockImplementation((_q: string, _topK?: number, signal?: AbortSignal) =>
      new Promise((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new ApiError('aborted', '已取消')))
      }))
    const w = mount(WorkbenchPage, { global: { plugins: [makeRouter()] } })
    await w.get('[data-test=mode-search]').trigger('click')
    await w.get('[data-test=input]').setValue('违约金过高怎么办')
    await w.get('[data-test=submit]').trigger('click')
    expect(w.find('[data-test=cancel]').exists()).toBe(true) // 检索中的取消按钮可见
    await w.get('[data-test=cancel]').trigger('click')
    await flushPromises()
    expect(searchMock).toHaveBeenCalledTimes(1)
    // 承重：去掉第三参（或不透传）时这里是 undefined——取消变成 no-op 必红
    expect(searchMock.mock.calls[0][2]?.aborted).toBe(true)
    expect(w.find('[data-test=search-result]').exists()).toBe(false)
    expect(w.find('[data-test=retry]').exists()).toBe(false) // 取消不是错误：不弹错误面板
    expect(w.find('[data-test=submit]').exists()).toBe(true) // loading 已复位
  })

  it('会话失效（专属路由 404 归型 auth-expired）→ 清 token 跳登录，不渲染错误面板', async () => {
    vi.mocked(apiLogin).mockResolvedValue(
      { access_token: 't', token_type: 'bearer', role: 'lawyer', team_id: 'team-a' })
    await login('u', 'p')
    const router = makeRouter()
    await router.push('/')
    const w = mount(WorkbenchPage, { global: { plugins: [router] } })
    qaMock.mockRejectedValue(new ApiError('auth-expired', '登录状态已失效，请重新登录', { status: 404 }))
    await ask(w, '合同违约怎么办？')
    expect(session.token).toBeNull()
    expect(localStorage.getItem('fl_lawyer_session')).toBeNull()
    expect(router.currentRoute.value.path).toBe('/login')
    // 会话失效不是普通错误：不弹 ErrorPanel（换了新错误面板会误导用户「重试」）
    expect(w.find('[data-test=error]').exists()).toBe(false)
  })

  it('429：倒计时内提交置灰，倒计时结束复位（countdown-end 接线）', async () => {
    vi.useFakeTimers()
    qaMock.mockRejectedValue(
      new ApiError('api', '请求过于频繁', { status: 429, code: 'rate_limited', retryAfter: 2 }))
    const w = mount(WorkbenchPage, { global: { plugins: [makeRouter()] } })
    await w.get('[data-test=input]').setValue('合同违约怎么办？')
    await w.get('[data-test=submit]').trigger('click')
    // 假表下不能用 flushPromises（依赖真 setTimeout）：推进 0ms 让 rejection 落进 catch
    await vi.advanceTimersByTimeAsync(0)
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeDefined()
    expect(w.get('[data-test=countdown]').text()).toContain('2 秒后可重试')
    await vi.advanceTimersByTimeAsync(2000)
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeUndefined()
    w.unmount()
  })

  it('429 倒计时内点重试：重试成功即解锁，不等随卸载消失的倒计时（裁决 A/I1 镜像）', async () => {
    // 倒计时中点重试会卸载 ErrorPanel，countdown-end 随组件一起消失不再发；
    // 若 submit() 不在起点复位 rateLimited，这次重试成功之后提交按钮会永久
    // 置灰（只能刷新页面）。这条钉住「任何一次提交都重新开始限流状态机」——
    // 与 AskPage 的 I1 用例同构，Workbench 不得只抄一半
    vi.useFakeTimers()
    qaMock
      .mockRejectedValueOnce(new ApiError('api', '请求过于频繁', { status: 429, code: 'rate_limited', retryAfter: 30 }))
      .mockResolvedValueOnce(answer({ answer: '重试成功的回答。' }))
    const w = mount(WorkbenchPage, { global: { plugins: [makeRouter()] } })
    await w.get('[data-test=input]').setValue('合同违约怎么办？')
    await w.get('[data-test=submit]').trigger('click')
    await vi.advanceTimersByTimeAsync(0)
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeDefined()
    // 还剩 30 秒时点重试：这一下把正在倒计时的 ErrorPanel 卸载掉，
    // 旧倒计时不会再走到 0，也就不会有任何 countdown-end
    await w.get('[data-test=retry]').trigger('click')
    await vi.advanceTimersByTimeAsync(0)
    expect(w.get('[data-test=result]').text()).toContain('重试成功的回答。')
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeUndefined()
    // 再推进 45 秒（远超 Retry-After）仍可提交：解锁只能来自本次提交的复位，
    // 而不是某个稍后才可能到达的 countdown-end（那条路已被卸载切断）
    await vi.advanceTimersByTimeAsync(45000)
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeUndefined()
    w.unmount()
  })
})

// 顶部身份与退出（设计 §四「顶部含当前角色/团队与退出（清 token）」）——Task 6
// 首审 m2：身份三元组换成静态占位、onLogout 清空两刀全绿，这两条钉住接线本身
describe('WorkbenchPage 顶部身份与退出', () => {
  it('顶部显示 session 的 username/role/teamId', () => {
    session.username = 'u-w'
    session.role = 'partner'
    session.teamId = 'team-w'
    const w = mount(WorkbenchPage, { global: { plugins: [makeRouter()] } })
    const identity = w.get('[data-test=identity]').text()
    expect(identity).toContain('u-w')
    expect(identity).toContain('partner')
    expect(identity).toContain('team-w')
  })

  it('点退出 → 清 session 与 localStorage 并跳登录页', async () => {
    session.token = 'tok-w'
    localStorage.setItem('fl_lawyer_session', JSON.stringify({
      token: 'tok-w', role: 'lawyer', teamId: 'team-w', username: 'u-w',
    }))
    const router = makeRouter()
    await router.push('/')
    const w = mount(WorkbenchPage, { global: { plugins: [router] } })
    await w.get('[data-test=logout]').trigger('click')
    await flushPromises()
    expect(session.token).toBeNull()
    expect(localStorage.getItem('fl_lawyer_session')).toBeNull()
    expect(router.currentRoute.value.path).toBe('/login')
  })
})
