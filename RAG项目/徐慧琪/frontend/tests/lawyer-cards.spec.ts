// 律师卡是演示数据的出口：字段一个不丢，且 demo=true 必须带「示例数据」角标
// （风险 11：不可构成虚假宣传）。两向断言——角标的有无只认 demo 字段本身。
import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import LawyerCards from '@/components/LawyerCards.vue'

const CARD = {
  name: '张文', org: '明理正衡律师事务所', field: '合同纠纷',
  contact_hint: '通过平台留言获取', demo: true,
}

describe('推荐律师卡（FR-9.3）', () => {
  it('字段齐全时逐项渲染，demo=true 带「示例数据」角标', () => {
    const wrapper = mount(LawyerCards, { props: { lawyers: [{ ...CARD }] } })
    const text = wrapper.text()
    for (const value of [CARD.name, CARD.org, CARD.field, CARD.contact_hint]) {
      expect(text).toContain(value)
    }
    expect(text).toContain('示例数据')
  })

  it('demo=false 不出现角标', () => {
    const wrapper = mount(LawyerCards, { props: { lawyers: [{ ...CARD, demo: false }] } })
    expect(wrapper.text()).toContain(CARD.name)
    expect(wrapper.text()).not.toContain('示例数据')
  })

  it('空列表不渲染卡片区', () => {
    const wrapper = mount(LawyerCards, { props: { lawyers: [] } })
    expect(wrapper.text()).toBe('')
  })
})
