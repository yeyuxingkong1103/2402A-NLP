// 费用卡是 FR-9.4 / AC-21 的展示落点：区间必须带单位、固定提示必须与区间同屏；
// 「没有依据」与「故障」的文案必须不同（③b 红线：故障 ≠ 没有依据）。
// 断言锚字面量——文案是需求本身。
import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import FeeCard from '@/components/FeeCard.vue'
import { FEE_DISCLAIMER } from '@/core/config'
import type { FeeRange } from '@/core/api/public'

function makeFee(patch: Partial<FeeRange> = {}): FeeRange {
  return {
    status: 'ok', low: 2, high: 50, unit: '万元', charge_basis: null,
    basis: null, source_doc: null, source_no: null, ...patch,
  }
}

describe('费用区间卡（FR-9.4 / AC-21）', () => {
  it('区间：两个端点与单位都在，固定提示逐字出现', () => {
    const wrapper = mount(FeeCard, { props: { fee: makeFee() } })
    const text = wrapper.text()
    expect(text).toContain('2')
    expect(text).toContain('50')
    expect(text).toContain('万元')
    expect(text).toContain(FEE_DISCLAIMER)
  })

  it('计价基础附在区间行上（N10：丢掉它会把计时收费读成一次性收费）', () => {
    const wrapper = mount(FeeCard, { props: { fee: makeFee({ charge_basis: '小时' }) } })
    expect(wrapper.text()).toContain('小时')
  })

  it('端点缺失：显示状态文字而非区间，固定提示仍在', () => {
    const wrapper = mount(FeeCard, {
      props: { fee: makeFee({ status: 'no_corpus', low: null, high: null, unit: null }) },
    })
    expect(wrapper.text()).toContain('暂无费用口径依据')
    expect(wrapper.text()).toContain(FEE_DISCLAIMER)
  })

  it('故障态文案与「无依据」不同（故障 ≠ 没有依据）', () => {
    const wrapper = mount(FeeCard, {
      props: { fee: makeFee({ status: 'unavailable', low: null, high: null, unit: null }) },
    })
    expect(wrapper.text()).toContain('费用信息暂时不可用')
    expect(wrapper.text()).not.toContain('暂无费用口径依据')
    expect(wrapper.text()).toContain(FEE_DISCLAIMER)
  })

  it('fee=null 整卡为空（连固定提示都不出现）', () => {
    const wrapper = mount(FeeCard, { props: { fee: null } })
    expect(wrapper.text()).toBe('')
    expect(wrapper.find('.fee-card').exists()).toBe(false)
  })
})
