// 结果卡是 FR-4.4 的落点：解读与原文必须分区且标签同时存在；公众/律师两种
// 响应形状的渲染差（律师卡与费用卡只属于公众侧）按**键在不在**判；正文只走
// 文本插值——把 answer 设成标记文本仍须显示为字面量（渲染方式即 XSS 防线）。
import { describe, expect, it } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createMemoryHistory, createRouter } from 'vue-router'
import AnswerCard from '@/components/AnswerCard.vue'
import FeeCard from '@/components/FeeCard.vue'
import LawyerCards from '@/components/LawyerCards.vue'
import { routes } from '@/public/router'
import type { PublicQAAnswer } from '@/core/api/public'
import type { QAAnswer } from '@/core/api/lawyer'

const ANSWER = '可以主张违约责任。'
const QUOTE = '当事人一方不履行合同义务…'

// 律师侧 7 键形状（schemas.py：**没有** lawyers / fee_range 两键）
function lawyerResult(patch: Partial<QAAnswer> = {}): QAAnswer {
  return {
    request_id: 'r-1',
    status: 'ok',
    answer: ANSWER,
    citations: [{ article: '584', paragraph: null, item: null, quote: QUOTE }],
    sources: [{ article_no: 584, path: '第三编 合同 / 第九章 买卖合同', source: 'hybrid', rerank_score: 0.9 }],
    disclaimer: '本回答仅供参考，不构成法律意见。',
    failures: [],
    ...patch,
  }
}

// 公众侧 11 键形状 = 7 键 + cause / field / lawyers / fee_range（四键恒存在）
function publicResult(patch: Partial<PublicQAAnswer> = {}): PublicQAAnswer {
  return {
    ...lawyerResult(),
    cause: '买卖合同纠纷',
    field: '合同纠纷',
    lawyers: [{ name: '张文', org: '明理正衡律师事务所', field: '合同纠纷',
      contact_hint: '通过平台留言获取', demo: true }],
    fee_range: { status: 'ok', low: 2, high: 50, unit: '万元', charge_basis: null,
      basis: null, source_doc: null, source_no: null },
    ...patch,
  }
}

// 引用链接是 RouterLink（终审 I1）：每次挂载都装真实公众路由表，让链接能在
// 应用内解析；未装路由器时 RouterLink 渲染不出 href，判据会失真
function mountCard(props: { result: QAAnswer | PublicQAAnswer; copyable?: boolean }) {
  return mount(AnswerCard, {
    props,
    global: { plugins: [createRouter({ history: createMemoryHistory(), routes })] },
  })
}

describe('问答结果卡（FR-4.4 / FR-8）', () => {
  it('律师侧形状：解读/原文/参考块都有，且没有公众区块', async () => {
    const router = createRouter({ history: createMemoryHistory(), routes })
    const wrapper = mount(AnswerCard, { props: { result: lawyerResult() }, global: { plugins: [router] } })
    const text = wrapper.text()
    expect(text).toContain('AI 解读，仅供参考')
    expect(text).toContain(ANSWER)
    expect(text).toContain('第584条')
    expect(text).toContain(QUOTE)
    // 引用跳转目标是字面量 URL（判据 5）：条号值直接进路径
    expect(wrapper.get('.citation-list__link').attributes('href'))
      .toBe('/law/minfadian/articles/584')
    // 承重（终审 I1）：链接必须是 RouterLink——退回普通 <a> 时点击不驱动
    // vue-router，这一击后 currentRoute 不动（浏览器里则是整页跳去公众入口）
    await wrapper.get('.citation-list__link').trigger('click')
    await flushPromises()
    expect(router.currentRoute.value.path).toBe('/law/minfadian/articles/584')
    expect(text).toContain('第三编 合同 / 第九章 买卖合同')
    // 律师侧 7 键里没有 lawyers / fee_range——两块卡整体不出现
    expect(wrapper.findComponent(LawyerCards).exists()).toBe(false)
    expect(wrapper.findComponent(FeeCard).exists()).toBe(false)
    expect(text).not.toContain('示例数据')
    expect(text).not.toContain('费用区间')
  })

  it('公众侧形状：律师卡与费用卡（含示例角标与免责）都渲染', () => {
    const wrapper = mountCard({ result: publicResult() })
    expect(wrapper.findComponent(LawyerCards).exists()).toBe(true)
    expect(wrapper.findComponent(FeeCard).exists()).toBe(true)
    const text = wrapper.text()
    expect(text).toContain('张文')
    expect(text).toContain('示例数据')
    expect(text).toContain('万元')
  })

  it('正文 pre-wrap 纯文本：<b> 显示为字面量而不是 DOM（XSS 防线）', () => {
    const wrapper = mountCard({ result: lawyerResult({ answer: '<b>粗体</b>' }) })
    expect(wrapper.text()).toContain('<b>粗体</b>')
    expect(wrapper.find('b').exists()).toBe(false)
    const body = wrapper.get('.answer-card__body')
    expect((body.element as HTMLElement).style.whiteSpace).toBe('pre-wrap')
  })

  it('disclaimer：空串不渲染，非空必现（逐字）', () => {
    const empty = mountCard({ result: lawyerResult({ disclaimer: '' }) })
    expect(empty.find('.answer-card__disclaimer').exists()).toBe(false)
    const filled = mountCard({
      result: lawyerResult({ disclaimer: '本回答仅供参考，不构成法律意见。' }),
    })
    expect(filled.get('.answer-card__disclaimer').text()).toBe('本回答仅供参考，不构成法律意见。')
  })

  it('need_more_info：显示单轮提示（两向）', () => {
    const need = mountCard({ result: publicResult({ status: 'need_more_info' }) })
    expect(need.text()).toContain('补充后可再问（每次提问独立）')
    const ok = mountCard({ result: publicResult() })
    expect(ok.text()).not.toContain('补充后可再问（每次提问独立）')
  })

  it('copyable 时按钮 emit buildCopyText 的字面量结果', async () => {
    const wrapper = mountCard({ result: publicResult(), copyable: true })
    await wrapper.get('.answer-card__copy').trigger('click')
    expect(wrapper.emitted('copied')).toEqual([[
      `${ANSWER}\n\n《中华人民共和国民法典》第584条：“${QUOTE}”`,
    ]])
    const readOnly = mountCard({ result: publicResult() })
    expect(readOnly.find('.answer-card__copy').exists()).toBe(false)
  })
})
