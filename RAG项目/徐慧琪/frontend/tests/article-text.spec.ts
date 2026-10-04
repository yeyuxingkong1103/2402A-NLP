// 法条原文卡的三件套：中文条号 / 正文 / 效力徽标，外加面包屑 path。
// 正文必须 pre-wrap 纯文本（设计 §六 禁 v-html）；复制按钮只 emit —— 写剪贴板
// 是页面副作用（裁决 2），组件测试只断言事件与字面量。
import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import ArticleText from '@/components/ArticleText.vue'

const ARTICLE = {
  request_id: 'r-1',
  article_no: 584,
  article_no_cn: '第五百八十四条',
  path: '第三编 合同 / 第二分编 典型合同 / 第九章 买卖合同',
  text: '当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。',
  status: '现行有效',
  law_id: 'minfadian',
}

describe('法条原文卡（FR-7.2）', () => {
  it('中文条号 / 正文 / 路径 / 效力徽标都在', () => {
    const wrapper = mount(ArticleText, { props: { article: ARTICLE } })
    const text = wrapper.text()
    expect(text).toContain('第五百八十四条')
    expect(text).toContain(ARTICLE.text)
    expect(text).toContain(ARTICLE.path)
    expect(text).toContain('现行有效')
  })

  it('正文是 pre-wrap 纯文本：标记原样显示、不变成 DOM', () => {
    const wrapper = mount(ArticleText, {
      props: { article: { ...ARTICLE, text: '<b>粗体</b>' } },
    })
    const body = wrapper.get('.article-text__body')
    // VTU 把 element 宽化成 VueNode<Element>，取 style 要显式落到 HTMLElement
    expect((body.element as HTMLElement).style.whiteSpace).toBe('pre-wrap')
    expect(body.text()).toBe('<b>粗体</b>')
    expect(wrapper.find('b').exists()).toBe(false)
  })

  it('copyable 时按钮 emit 规范引用（字面量），不带剪贴板副作用', async () => {
    const wrapper = mount(ArticleText, { props: { article: ARTICLE, copyable: true } })
    await wrapper.get('.article-text__copy').trigger('click')
    expect(wrapper.emitted('copied')).toEqual([[
      `《中华人民共和国民法典》第584条：“${ARTICLE.text}”`,
    ]])
  })

  it('默认不可复制：没有按钮', () => {
    const wrapper = mount(ArticleText, { props: { article: ARTICLE } })
    expect(wrapper.find('.article-text__copy').exists()).toBe(false)
  })
})
