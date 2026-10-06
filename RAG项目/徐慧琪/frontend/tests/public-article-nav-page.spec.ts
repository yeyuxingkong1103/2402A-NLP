// NavPage / ArticlePage：一个把「点条号」变成真实路由跳转（用真 router +
// memory history，断的是真实跳转而不是 mock 出来的 push 调用），一个把路由
// 参数变成条文请求与展示。API 全 mock。
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createMemoryHistory, createRouter } from 'vue-router'
import type { Router } from 'vue-router'
import ArticlePage from '@/public/pages/ArticlePage.vue'
import NavPage from '@/public/pages/NavPage.vue'
import { ApiError } from '@/core/api/request'
import { article, nav } from '@/core/api/public'
import { routes } from '@/public/router'
import type { ArticleResponse, NavResponse } from '@/core/api/public'

vi.mock('@/core/api/public', () => ({ publicQa: vi.fn(), article: vi.fn(), nav: vi.fn() }))

const articleMock = vi.mocked(article)
const navMock = vi.mocked(nav)

// 被挂载的是页面组件本身，但路由是真实实例（routes 与生产同一张表）——
// 跳转目标若与生产路由表不一致，push 会落到「无匹配」而不是悄悄放过
function makeRouter(): Router {
  return createRouter({ history: createMemoryHistory(), routes })
}

// 两层足够：交互判的是「叶 → 跳转参数」，层级数由 nav.spec 另判
const NAV: NavResponse = {
  request_id: 'r-nav',
  laws: [{
    law_id: 'minfadian',
    nodes: [{
      title: '第一编 总则', level: 0, path: '第一编 总则',
      articles: [{ article_no: 584, article_no_cn: '第五百八十四条' }],
      children: [],
    }],
  }],
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

// jsdom 没有 navigator.clipboard：复制判据装替身（终审 I2）；afterEach 摘掉，
// 不把替身漏给别的用例
function stubClipboard() {
  const writeText = vi.fn().mockResolvedValue(undefined)
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
  return writeText
}

beforeEach(() => {
  articleMock.mockReset()
  navMock.mockReset()
})

afterEach(() => Reflect.deleteProperty(navigator, 'clipboard'))

describe('NavPage', () => {
  it('点文章叶 → 真实跳转到 /law/minfadian/articles/584（law_id 来自数据，不硬编码）', async () => {
    navMock.mockResolvedValue(NAV)
    const router = makeRouter()
    const w = mount(NavPage, { global: { plugins: [router] } })
    await flushPromises()
    await contentOf(w, '第一编 总则').find('.el-tree-node__expand-icon').trigger('click')
    await flushPromises()
    await contentOf(w, '第五百八十四条').trigger('click')
    await flushPromises()
    expect(router.currentRoute.value.path).toBe('/law/minfadian/articles/584')
    expect(router.currentRoute.value.params).toMatchObject({ lawId: 'minfadian', no: '584' })
  })

  it('错误 → ErrorPanel 透传后端 message，复制 request_id 进剪贴板（终审 I2）', async () => {
    const writeText = stubClipboard()
    navMock.mockRejectedValue(
      new ApiError('api', '服务暂时不可用', { status: 503, requestId: 'rid-nav' }))
    const router = makeRouter()
    const w = mount(NavPage, { global: { plugins: [router] } })
    await flushPromises()
    expect(w.get('.error-panel__message').text()).toContain('服务暂时不可用')
    await w.get('[data-test=copy-id]').trigger('click')
    // 承重：页面去掉 @copied 后点击是静默 no-op，这条必红
    expect(writeText).toHaveBeenCalledWith('rid-nav')
  })
})

describe('ArticlePage', () => {
  it('按路由参数取文，显示原文与 path', async () => {
    articleMock.mockResolvedValue(articleFixture())
    const router = makeRouter()
    await router.push('/law/minfadian/articles/584')
    await router.isReady()
    const w = mount(ArticlePage, { global: { plugins: [router] } })
    await flushPromises()
    // 路径参数原样传给 API（都是字符串，不做数字转换——条号不含前导零问题）
    expect(articleMock).toHaveBeenCalledWith('minfadian', '584')
    expect(w.text()).toContain('第五百八十四条')
    expect(w.text()).toContain('当事人一方不履行合同义务')
    expect(w.text()).toContain('第三编 合同 > 第一分编 通则 > 第四章 合同的履行')
  })

  it('同一实例内换条号（/articles/1 → /articles/2）重取并显示新正文', async () => {
    // 路由复用同一组件实例，只在挂载时取一次的话第二篇永远显示第一篇
    // （静默错页）；这条让「watch 路由参数」成为承重判据
    articleMock
      .mockResolvedValueOnce(articleFixture({ article_no: 1, article_no_cn: '第一条', text: '第一条的正文。' }))
      .mockResolvedValueOnce(articleFixture({ article_no: 2, article_no_cn: '第二条', text: '第二条的正文。' }))
    const router = makeRouter()
    await router.push('/law/minfadian/articles/1')
    await router.isReady()
    const w = mount(ArticlePage, { global: { plugins: [router] } })
    await flushPromises()
    expect(w.text()).toContain('第一条的正文。')
    await router.push('/law/minfadian/articles/2')
    await flushPromises()
    expect(articleMock).toHaveBeenCalledTimes(2)
    expect(w.text()).toContain('第二条的正文。')
  })

  it('错误 → ErrorPanel 透传后端 message，复制 request_id 进剪贴板（终审 I2）', async () => {
    const writeText = stubClipboard()
    articleMock.mockRejectedValue(
      new ApiError('api', '服务暂时不可用', { status: 503, requestId: 'rid-art' }))
    const router = makeRouter()
    await router.push('/law/minfadian/articles/584')
    await router.isReady()
    const w = mount(ArticlePage, { global: { plugins: [router] } })
    await flushPromises()
    expect(w.get('.error-panel__message').text()).toContain('服务暂时不可用')
    await w.get('[data-test=copy-id]').trigger('click')
    // 承重：页面去掉 @copied 后点击是静默 no-op，这条必红
    expect(writeText).toHaveBeenCalledWith('rid-art')
  })
})
