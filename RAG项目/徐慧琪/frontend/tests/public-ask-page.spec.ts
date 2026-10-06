// AskPage 是公众入口的主页：提问链路（QuestionBox → publicQa → 结果卡）与
// 四种「没有正常回答」的形态（need_more_info / 错误 / 取消 / 429 等待）都在
// 这里钉住。API 全 mock——页面测试判的是**接线**，真实的网络与错误归型语义
// 在 api-request / api-endpoints spec 里另判。
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createMemoryHistory, createRouter } from 'vue-router'
import AskPage from '@/public/pages/AskPage.vue'
import { routes } from '@/public/router'
import { ApiError } from '@/core/api/request'
import { publicQa } from '@/core/api/public'
import type { PublicQAAnswer } from '@/core/api/public'

// 结果卡的引用链接是 RouterLink（终审 I1）：AskPage 的每次挂载都装真实公众
// 路由表（memory history），未装路由器时 RouterLink 解析失败会让渲染形态失真
function makeRouter() {
  return createRouter({ history: createMemoryHistory(), routes })
}

// 工厂式 mock：页面只消费 publicQa；article/nav 一并给出，避免该模块被别的
// 页面经同一 spec 文件间接 import 时拿到 undefined
vi.mock('@/core/api/public', () => ({ publicQa: vi.fn(), article: vi.fn(), nav: vi.fn() }))

const publicQaMock = vi.mocked(publicQa)

function answer(over: Partial<PublicQAAnswer> = {}): PublicQAAnswer {
  return {
    request_id: 'r-1',
    status: 'ok',
    answer: '可以退租，但需看合同约定。',
    citations: [{ article: '584', paragraph: null, item: null, quote: '当事人一方不履行合同义务…' }],
    sources: [],
    disclaimer: '本回答仅为一般性法律信息，不构成正式法律意见，具体案件请咨询律师。',
    failures: [],
    cause: null,
    field: null,
    lawyers: [{ name: '张三', org: '示例律师事务所', field: '合同纠纷', contact_hint: '010-00000000', demo: true }],
    fee_range: {
      status: 'ok', low: 1000, high: 8000, unit: '元/小时', charge_basis: '按小时计费',
      basis: null, source_doc: null, source_no: null,
    },
    ...over,
  }
}

async function ask(w: ReturnType<typeof mount>, text: string) {
  await w.get('[data-test=input]').setValue(text)
  await w.get('[data-test=submit]').trigger('click')
  await flushPromises()
}

// 429 用例中途换假表；不还原会把后续文件里的真实等待全冻住
afterEach(() => vi.useRealTimers())

beforeEach(() => {
  publicQaMock.mockReset()
})

describe('AskPage（公众问答）', () => {
  it('提交后出结果卡：解读标签 + 律师卡 + 费用卡 + 免责逐答必现', async () => {
    publicQaMock.mockResolvedValue(answer())
    const w = mount(AskPage, { global: { plugins: [makeRouter()] } })
    await ask(w, '  合同违约怎么办？  ')
    // QuestionBox 的 trim 协议在页面这一层仍然成立（mock 收到的是 trim 后问句）
    expect(publicQaMock).toHaveBeenCalledTimes(1)
    expect(publicQaMock.mock.calls[0][0]).toBe('合同违约怎么办？')
    const card = w.get('[data-test=result]')
    expect(card.text()).toContain('AI 解读，仅供参考')
    expect(card.text()).toContain('张三') // 律师卡（demo 数据带「示例数据」角标）
    expect(card.text()).toContain('示例数据')
    expect(card.text()).toContain('费用区间')
    expect(card.text()).toContain('参考区间，不构成报价或委托')
    expect(card.text()).toContain('本回答仅为一般性法律信息') // 免责来自 API 字段，逐字透出
  })

  it('两次提交 → 两张独立结果卡，最新在上（内存堆叠，不互相覆盖）', async () => {
    publicQaMock
      .mockResolvedValueOnce(answer({ answer: '第一次的回答。' }))
      .mockResolvedValueOnce(answer({ answer: '第二次的回答。' }))
    const w = mount(AskPage, { global: { plugins: [makeRouter()] } })
    await ask(w, '第一问')
    await ask(w, '第二问')
    const cards = w.findAll('[data-test=result]')
    expect(cards).toHaveLength(2)
    expect(cards[0].text()).toContain('第二次的回答。')
    expect(cards[1].text()).toContain('第一次的回答。')
  })

  it('need_more_info：显示补充提示（补充后可再问，每次提问独立）', async () => {
    publicQaMock.mockResolvedValue(answer({ status: 'need_more_info', answer: '请补充合同签订时间。' }))
    const w = mount(AskPage, { global: { plugins: [makeRouter()] } })
    await ask(w, '我该怎么办？')
    expect(w.get('[data-test=result]').text()).toContain('补充后可再问（每次提问独立）')
  })

  it('错误 → ErrorPanel：后端 message 与 request_id 可见，且不产生结果卡', async () => {
    publicQaMock.mockRejectedValue(
      new ApiError('api', '服务暂时不可用', { status: 503, code: 'service_unavailable', requestId: 'rid-9' }))
    const w = mount(AskPage, { global: { plugins: [makeRouter()] } })
    await ask(w, '合同违约怎么办？')
    expect(w.find('[data-test=result]').exists()).toBe(false)
    expect(w.text()).toContain('服务暂时不可用')
    expect(w.text()).toContain('rid-9')
    expect(w.find('[data-test=retry]').exists()).toBe(true)
  })

  it('ErrorPanel 的重试重发最后一问（不是用户改了一半的输入框）', async () => {
    publicQaMock
      .mockRejectedValueOnce(new ApiError('api', '服务暂时不可用', { status: 503 }))
      .mockResolvedValueOnce(answer({ answer: '重试后的回答。' }))
    const w = mount(AskPage, { global: { plugins: [makeRouter()] } })
    await ask(w, '合同违约怎么办？')
    await w.get('[data-test=retry]').trigger('click')
    await flushPromises()
    expect(publicQaMock).toHaveBeenCalledTimes(2)
    expect(publicQaMock.mock.calls[1][0]).toBe('合同违约怎么办？')
    expect(w.find('[data-test=retry]').exists()).toBe(false)
    expect(w.get('[data-test=result]').text()).toContain('重试后的回答。')
  })

  it('取消 → 静默：不出结果卡、不报错，输入框内容保留、回到可提交态', async () => {
    // mock 复刻 request.ts 的取消归型：signal 中止 → ApiError(aborted)。
    // 页面必须把它与错误区分开（设计 §五「取消不报错」）
    publicQaMock.mockImplementation((_q: string, signal?: AbortSignal) =>
      new Promise((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new ApiError('aborted', '已取消')))
      }))
    const w = mount(AskPage, { global: { plugins: [makeRouter()] } })
    await w.get('[data-test=input]').setValue('合同违约怎么办？')
    await w.get('[data-test=submit]').trigger('click')
    expect(w.find('[data-test=submit]').exists()).toBe(false) // 提交中：只剩取消
    await w.get('[data-test=cancel]').trigger('click')
    await flushPromises()
    expect(w.find('[data-test=result]').exists()).toBe(false)
    expect(w.find('[data-test=retry]').exists()).toBe(false) // 取消不是错误：不弹错误面板
    expect(w.find('[data-test=submit]').exists()).toBe(true) // loading 已复位
    expect((w.get('[data-test=input]').element as HTMLTextAreaElement).value).toBe('合同违约怎么办？')
  })

  it('429：倒计时内提交置灰，倒计时结束复位（裁决 A 的页面接线）', async () => {
    vi.useFakeTimers()
    publicQaMock.mockRejectedValue(
      new ApiError('api', '请求过于频繁', { status: 429, code: 'rate_limited', retryAfter: 2 }))
    const w = mount(AskPage, { global: { plugins: [makeRouter()] } })
    await w.get('[data-test=input]').setValue('合同违约怎么办？')
    await w.get('[data-test=submit]').trigger('click')
    // 假表下不能用 flushPromises（它内部依赖真 setTimeout）：用 0ms 推进把
    // mock 的 rejection 落进 catch
    await vi.advanceTimersByTimeAsync(0)
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeDefined()
    expect(w.get('[data-test=countdown]').text()).toContain('2 秒后可重试')
    await vi.advanceTimersByTimeAsync(2000)
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeUndefined()
    w.unmount()
    vi.useRealTimers()
  })

  it('429 倒计时内点重试：重试成功即解锁，不等随卸载消失的倒计时（I1）', async () => {
    // 倒计时中点重试会卸载 ErrorPanel，countdown-end 随组件一起消失不再发；
    // 若 submit() 不在起点复位 rateLimited，这次重试成功之后提交按钮会永久
    // 置灰（只能刷新页面）。这条钉住「任何一次提交都重新开始限流状态机」
    vi.useFakeTimers()
    publicQaMock
      .mockRejectedValueOnce(new ApiError('api', '请求过于频繁', { status: 429, code: 'rate_limited', retryAfter: 30 }))
      .mockResolvedValueOnce(answer({ answer: '重试成功的回答。' }))
    const w = mount(AskPage, { global: { plugins: [makeRouter()] } })
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
    vi.useRealTimers()
  })
})
