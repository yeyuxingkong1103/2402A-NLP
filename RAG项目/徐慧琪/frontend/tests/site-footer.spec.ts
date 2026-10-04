// FR-8.4 的页脚常驻位：免责声明不依赖某次回答的 disclaimer 字段——两个入口的
// 每一页都要有这句话；「法条浏览」链接是公众/律师侧共用的导航入口。
//
// 终审 I1：「法条浏览」必须是 RouterLink。普通 a 是整页导航，律师用户会被送出
// 律师壳进入公众入口；判别不能看标签名（RouterLink 渲染出来同样是 <a>），要判
// **点击是否驱动本入口的路由实例**——普通 a 在 jsdom 里不驱动 vue-router，点击后
// currentRoute 不动（浏览器里则整页跳走）。本 spec 用真实律师路由表挂载，
// /nav 是它的成员路由，承接「留在律师入口内」这条意图。
import { afterEach, describe, expect, it } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createMemoryHistory } from 'vue-router'
import SiteFooter from '@/components/SiteFooter.vue'
import { createLawyerRouter } from '@/lawyer/router'
import { logout, session } from '@/core/useAuth'

afterEach(() => { localStorage.clear(); logout() })

describe('SiteFooter', () => {
  it('免责文案逐字常驻（含「不构成正式法律意见」）', () => {
    const w = mount(SiteFooter, { global: { plugins: [createLawyerRouter(createMemoryHistory())] } })
    expect(w.get('.site-footer__disclaimer').text())
      .toBe('本系统输出仅为一般性法律信息，不构成正式法律意见，具体案件请咨询律师。')
  })

  it('法条浏览是 RouterLink：href=/nav 且点击驱动本入口路由（退回普通 a 必红）', async () => {
    session.token = 'tok-footer' // 过守卫；本用例只判「点击是否驱动本路由实例」
    const router = createLawyerRouter(createMemoryHistory())
    const w = mount(SiteFooter, { global: { plugins: [router] } })
    const link = w.get('.site-footer__nav')
    expect(link.text()).toBe('法条浏览')
    expect(link.attributes('href')).toBe('/nav')
    // 承重：普通 <a href="/nav"> 不驱动 vue-router，这一击后 currentRoute 仍是
    // 起始路径——正是 I1 的失效形态（整页导航离开当前入口）
    await link.trigger('click')
    await flushPromises()
    expect(router.currentRoute.value.path).toBe('/nav')
  })
})
