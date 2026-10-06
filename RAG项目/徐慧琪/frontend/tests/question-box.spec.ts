// 提问框是用户输入进系统的唯一入口：上限、空白、trim 三个边界必须本地拦住
// （后端仍会再判，但 400 来回一趟对用户是白等）；loading 的计时与 interval
// 清理也在这里钉——QA 实测一发 13~24s，用户需要知道系统还在跑。
import { afterEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import QuestionBox from '@/components/QuestionBox.vue'
import { MAX_QUESTION_CHARS } from '@/core/config'

// 定时器用例结束后必须还原，否则后续文件里的真实等待全被冻住
afterEach(() => vi.useRealTimers())

describe('QuestionBox', () => {
  it(`超过 ${MAX_QUESTION_CHARS} 字：禁提交且给出超限提示；回到上限内可提交`, async () => {
    const w = mount(QuestionBox)
    // DOM 的 maxlength 是浏览器层的键盘输入上限；setValue 绕过它（模拟粘贴/
    // 输入法组合），禁提交则由 JS 逻辑兜底——两层都要与常量同值（改绑 501 必须变红）
    expect(w.get('[data-test=input]').attributes('maxlength')).toBe(String(MAX_QUESTION_CHARS))
    await w.get('[data-test=input]').setValue('长'.repeat(MAX_QUESTION_CHARS + 1))
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeDefined()
    expect(w.text()).toContain(`已超出 ${MAX_QUESTION_CHARS} 字上限`)
    // 边界精确：501 红、500 绿（差一位就等于没有上限）
    await w.get('[data-test=input]').setValue('长'.repeat(MAX_QUESTION_CHARS))
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeUndefined()
    // 负向分支：未超限时不渲染超限提示（恒真渲染会让 500 字也报「已超出」）
    expect(w.find('[data-test=over-limit]').exists()).toBe(false)
  })

  it('全空白禁提交（不禁的话后端收到的是一串空格）', async () => {
    const w = mount(QuestionBox)
    await w.get('[data-test=input]').setValue('   \n\t  ')
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeDefined()
    expect(w.emitted('submit')).toBeUndefined()
  })

  it('emit 的载荷是 trim 后的问句', async () => {
    const w = mount(QuestionBox)
    await w.get('[data-test=input]').setValue('  合同违约怎么办？  ')
    await w.get('[data-test=submit]').trigger('click')
    expect(w.emitted('submit')).toEqual([['合同违约怎么办？']])
  })

  it('loading：显示取消与「已等待 N 秒」，取消 emit cancel', async () => {
    vi.useFakeTimers()
    const w = mount(QuestionBox, { props: { loading: true } })
    expect(w.find('[data-test=submit]').exists()).toBe(false) // 在跑的请求不被第二次提交覆盖
    expect(w.get('[data-test=elapsed]').text()).toContain('已等待 0 秒')
    vi.advanceTimersByTime(2000)
    await w.vm.$nextTick()
    expect(w.get('[data-test=elapsed]').text()).toContain('已等待 2 秒')
    await w.get('[data-test=cancel]').trigger('click')
    expect(w.emitted('cancel')).toHaveLength(1)
  })

  it('卸载后计时器清零（删掉清理这条必须变红）', () => {
    // 裁决 3：loading 的 interval 是组件里唯一会在卸载后继续跑的东西——
    // 删掉 onUnmounted(stopTimer) 时这条必须在定时器计数上变红
    vi.useFakeTimers()
    const w = mount(QuestionBox, { props: { loading: true } })
    expect(vi.getTimerCount()).toBe(1)
    vi.advanceTimersByTime(1000)
    w.unmount()
    expect(vi.getTimerCount()).toBe(0)
    vi.useRealTimers()
  })

  it('disabled：提交可见但置灰，输入框可编辑，不出取消/计时（裁决 A）', async () => {
    // 429 倒计时内禁提交的组件面：置灰而不是消失（按钮还在，用户知道等完
    // 可以按）；也不能长成 loading 的形态（取消/计时是「请求在跑」的语义）
    const w = mount(QuestionBox, { props: { disabled: true } })
    await w.get('[data-test=input]').setValue('合同违约怎么办？')
    const submit = w.get('[data-test=submit]')
    expect(submit.attributes('disabled')).toBeDefined()
    expect(w.find('[data-test=cancel]').exists()).toBe(false)
    expect(w.find('[data-test=elapsed]').exists()).toBe(false)
    expect(w.get('[data-test=input]').attributes('disabled')).toBeUndefined()
    await submit.trigger('click')
    expect(w.emitted('submit')).toBeUndefined()
  })

  it('loading 优先于 disabled：请求在跑的形态不被置灰态顶掉（裁决 A）', () => {
    const w = mount(QuestionBox, { props: { loading: true, disabled: true } })
    expect(w.find('[data-test=cancel]').exists()).toBe(true)
    expect(w.get('[data-test=elapsed]').text()).toContain('已等待 0 秒')
    expect(w.find('[data-test=submit]').exists()).toBe(false)
  })

  it('disabled 为 false 时不改变既有可提交行为（负向分支）', async () => {
    // 恒真禁用会让正常提问永远按不下去；这条钉住 disabled=false 与不传等价
    const w = mount(QuestionBox, { props: { disabled: false } })
    await w.get('[data-test=input]').setValue('合同违约怎么办？')
    expect(w.get('[data-test=submit]').attributes('disabled')).toBeUndefined()
    await w.get('[data-test=submit]').trigger('click')
    expect(w.emitted('submit')).toEqual([['合同违约怎么办？']])
  })
})
