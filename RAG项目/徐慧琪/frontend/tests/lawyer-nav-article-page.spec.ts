// 律师侧 Nav/Article 是共享路由页的律师版（从 workbench-page.spec.ts 拆出：
// 原文件 282/300 非空行逼近闸门，终审 R17 要求补判据时另开新文件）。
// 跳转的 law_id 必须来自数据（裁决 D①），错误分支必须有判据（裁决 D②），
// ArticleText 的复制落点在页面；ErrorPanel 的复制 request_id 接线（终审 I2）
// 也在这里对两页各钉一向。
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createMemoryHistory } from 'vue-router'
import LawyerArticlePage from '@/lawyer/pages/ArticlePage.vue'
import LawyerNavPage from '@/lawyer/pages/NavPage.vue'
import { createLawyerRouter } from '@/lawyer/router'
import { ApiError } from '@/core/api/request'
import { logout, session } from '@/core/useAuth'
import { article, nav } from '@/core/api/public'
import type { ArticleResponse, NavResponse } from '@/core/api/public'

// 工厂式 mock：router.ts 会拉起全部六个页面，端点一并给出，避免未使用的页面
// import 到 undefined（vitest 对缺失导出直接报错）
vi.mock('@/core/api/lawyer', () => ({
  login: vi.fn(), qa: vi.fn(), search: vi.fn(), casesSearch: vi.fn(), exportAudit: vi.fn(),
}))
vi.mock('@/core/api/public', () => ({ publicQa: vi.fn(), article: vi.fn(), nav: vi.fn() }))

const navMock = vi.mocked(nav)
const articleMock = vi.mocked(article)

function makeRouter() {
  return createLawyerRouter(createMemoryHistory())
}

// law_id 刻意不是 'minfadian'（裁决 D①）：写死常量的实现会在这里红
function navFixture(lawId: string): NavResponse {
  return {
    request_id: 'r-nav',
    laws: [{
      law_id: lawId,
      nodes: [{
        title: '第一编 总则', level: 0, path: '第一编 总则',
        articles: [{ article_no: 584, article_no_cn: '第五百八十四条' }],
        children: [],
      }],
    }],
  }
}

function articleFixture(over: Partial<ArticleResponse> = {}): ArticleResponse {
  return {
    request_id: 'r-a',
    article_no: 584,
    article_no_cn: '第五百八十四条',
    path: '第三编 合同 > 第一分编 通则 > 第四章 合同的履行',
    text: '当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担违约责任。',
    status: '现行有效',
    law_id: 'minfadian',
    ...over,
  }
}

/** 按 label 精确找树内容行（同 nav-tree.spec 的手法） */
function contentOf(w: ReturnType<typeof mount>, label: string) {
  const hit = w.findAll('.el-tree-node__content').find((c) => c.text() === label)
  if (!hit) throw new Error(`未找到节点：${label}`)
  return hit
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
  navMock.mockReset()
  articleMock.mockReset()
})

afterEach(() => Reflect.deleteProperty(navigator, 'clipboard'))

describe('律师侧 NavPage / ArticlePage', () => {
  it('叶点击 → 用数据里的 law_id（hetongbian）跳详情，不写死 minfadian', async () => {
    navMock.mockResolvedValue(navFixture('hetongbian'))
    session.token = 'tok-nav' // 放行守卫；本用例只判跳转目标
    const router = makeRouter()
    const w = mount(LawyerNavPage, { global: { plugins: [router] } })
    await flushPromises()
    await contentOf(w, '第一编 总则').find('.el-tree-node__expand-icon').trigger('click')
    await flushPromises()
    await contentOf(w, '第五百八十四条').trigger('click')
    await flushPromises()
    expect(router.currentRoute.value.path).toBe('/law/hetongbian/articles/584')
    expect(router.currentRoute.value.params).toMatchObject({ lawId: 'hetongbian', no: '584' })
  })

  it('NavPage 错误 → ErrorPanel 渲染后端 message，重试重发', async () => {
    navMock.mockRejectedValue(new ApiError('api', '服务暂时不可用', { status: 503 }))
    const w = mount(LawyerNavPage, { global: { plugins: [makeRouter()] } })
    await flushPromises()
    expect(w.get('[data-test=error]').text()).toContain('服务暂时不可用')
    await w.get('[data-test=retry]').trigger('click')
    await flushPromises()
    expect(navMock).toHaveBeenCalledTimes(2)
  })

  it('NavPage 错误面板复制 → 剪贴板收到 request_id（终审 I2，去掉 @copied 必红）', async () => {
    const writeText = stubClipboard()
    navMock.mockRejectedValue(
      new ApiError('api', '服务暂时不可用', { status: 503, requestId: 'rid-nav' }))
    const w = mount(LawyerNavPage, { global: { plugins: [makeRouter()] } })
    await flushPromises()
    await w.get('[data-test=copy-id]').trigger('click')
    expect(writeText).toHaveBeenCalledWith('rid-nav')
  })

  it('ArticlePage 错误 → ErrorPanel 渲染后端 message，重试重发', async () => {
    articleMock.mockRejectedValue(new ApiError('api', '未找到', { status: 404 }))
    session.token = 'tok-a'
    const router = makeRouter()
    await router.push('/law/minfadian/articles/999')
    const w = mount(LawyerArticlePage, { global: { plugins: [router] } })
    await flushPromises()
    expect(w.get('[data-test=error]').text()).toContain('未找到')
    await w.get('[data-test=retry]').trigger('click')
    await flushPromises()
    expect(articleMock).toHaveBeenCalledTimes(2)
  })

  it('ArticlePage 错误面板复制 → 剪贴板收到 request_id（终审 I2，去掉 @copied 必红）', async () => {
    const writeText = stubClipboard()
    articleMock.mockRejectedValue(
      new ApiError('api', '服务暂时不可用', { status: 503, requestId: 'rid-art' }))
    session.token = 'tok-a'
    const router = makeRouter()
    await router.push('/law/minfadian/articles/584')
    const w = mount(LawyerArticlePage, { global: { plugins: [router] } })
    await flushPromises()
    await w.get('[data-test=copy-id]').trigger('click')
    expect(writeText).toHaveBeenCalledWith('rid-art')
  })

  it('ArticlePage 复制 → 剪贴板收到规范引用字面量（FR-7.4）', async () => {
    const writeText = stubClipboard()
    articleMock.mockResolvedValue(articleFixture())
    session.token = 'tok-a'
    const router = makeRouter()
    await router.push('/law/minfadian/articles/584')
    const w = mount(LawyerArticlePage, { global: { plugins: [router] } })
    await flushPromises()
    await w.get('.article-text__copy').trigger('click')
    expect(writeText).toHaveBeenCalledWith(
      '《中华人民共和国民法典》第584条：“当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担违约责任。”')
  })
})
