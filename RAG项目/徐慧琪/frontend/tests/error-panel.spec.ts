// request_id 是「按 id 回查运维日志」的唯一把手（交付说明 §四）——错误面板
// 必须显示它；429 的倒计时来自 Retry-After，不是前端自己编的。
import { describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { ApiError } from '@/core/api/request'
import ErrorPanel from '@/components/ErrorPanel.vue'

describe('ErrorPanel', () => {
  it('显示后端 message / code / request_id，并可重试', async () => {
    const err = new ApiError('api', '服务暂时不可用', { status: 503, code: 'service_unavailable', requestId: 'rid-9' })
    const w = mount(ErrorPanel, { props: { error: err } })
    expect(w.text()).toContain('服务暂时不可用')
    expect(w.text()).toContain('service_unavailable')
    expect(w.text()).toContain('rid-9')
    await w.get('[data-test=retry]').trigger('click')
    expect(w.emitted('retry')).toHaveLength(1)
  })

  it('复制按钮只 emit copied(request_id)：剪贴板是页面副作用（裁决 1）', async () => {
    const err = new ApiError('api', '服务暂时不可用', { status: 503, code: 'service_unavailable', requestId: 'rid-9' })
    const w = mount(ErrorPanel, { props: { error: err } })
    await w.get('[data-test=copy-id]').trigger('click')
    expect(w.emitted('copied')).toEqual([['rid-9']])
  })

  it('429 显示 Retry-After 倒计时', async () => {
    vi.useFakeTimers()
    const err = new ApiError('api', '请求过于频繁', { status: 429, code: 'rate_limited', retryAfter: 30 })
    const w = mount(ErrorPanel, { props: { error: err } })
    expect(w.text()).toContain('30')
    vi.advanceTimersByTime(1000)
    await w.vm.$nextTick()
    expect(w.text()).toContain('29')
    vi.useRealTimers()
    w.unmount() // 卸载必须清 interval，否则测试进程泄漏定时器
  })

  it('卸载后倒计时 interval 清零（删掉清理这条必须变红）', () => {
    // 裁决 3（控制者追加判据）：上面那条注释只是注释——「卸载清理」要有能从
    // 绿变红的判据，否则删掉 onUnmounted(stopTimer) 不会有任何测试报警
    vi.useFakeTimers()
    const err = new ApiError('api', '请求过于频繁', { status: 429, code: 'rate_limited', retryAfter: 5 })
    const w = mount(ErrorPanel, { props: { error: err } })
    expect(vi.getTimerCount()).toBe(1)
    vi.advanceTimersByTime(1000)
    w.unmount()
    expect(vi.getTimerCount()).toBe(0)
    vi.useRealTimers()
  })

  it('没有 Retry-After（retryAfter 为 null）不渲染倒计时', () => {
    // 负向分支：倒计时只在 Retry-After 存在时出现。恒真渲染会让 503 也显示
    // 「0 秒后可重试」，把用户指向一次并不存在的等待——必须能变红
    const err = new ApiError('api', '服务暂时不可用', { status: 503, code: 'service_unavailable' })
    const w = mount(ErrorPanel, { props: { error: err } })
    expect(w.find('[data-test=countdown]').exists()).toBe(false)
  })

  it('倒计时走到 0 恰好 emit 一次 countdown-end（裁决 A）', async () => {
    // 页面靠这个事件解除「倒计时内禁提交」：漏发 = 提交按钮永远禁用，
    // 重复发 = 反复复位；两者在页面上都没有别的红灯，只能在这里钉死次数
    vi.useFakeTimers()
    const err = new ApiError('api', '请求过于频繁', { status: 429, code: 'rate_limited', retryAfter: 2 })
    const w = mount(ErrorPanel, { props: { error: err } })
    vi.advanceTimersByTime(1000)
    await w.vm.$nextTick()
    expect(w.emitted('countdown-end')).toBeUndefined() // 还没到 0，不发
    vi.advanceTimersByTime(1000)
    await w.vm.$nextTick()
    expect(w.emitted('countdown-end')).toHaveLength(1)
    vi.advanceTimersByTime(5000) // 停表后再走时间也不重复发
    await w.vm.$nextTick()
    expect(w.emitted('countdown-end')).toHaveLength(1)
    w.unmount()
    vi.useRealTimers()
  })

  it('Retry-After: 0 挂载即 emit 一次 countdown-end（没有一秒可等）', () => {
    // 后端允许立刻重试：页面不该白锁一秒——事件应在挂载时同步发出
    const err = new ApiError('api', '请求过于频繁', { status: 429, code: 'rate_limited', retryAfter: 0 })
    const w = mount(ErrorPanel, { props: { error: err } })
    expect(w.text()).toContain('0 秒后可重试')
    expect(w.emitted('countdown-end')).toHaveLength(1)
  })

  it('没有 Retry-After 不发 countdown-end（没有倒计时就没有归零事件）', () => {
    // 反向：503 若也发 countdown-end，页面的 429 锁还未置位就被立刻复位
    const err = new ApiError('api', '服务暂时不可用', { status: 503 })
    const w = mount(ErrorPanel, { props: { error: err } })
    expect(w.emitted('countdown-end')).toBeUndefined()
  })
})
