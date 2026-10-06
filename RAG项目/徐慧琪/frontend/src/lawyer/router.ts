import { createRouter, createWebHashHistory } from 'vue-router'
import type { Router, RouteRecordRaw, RouterHistory } from 'vue-router'
import { session, logout } from '@/core/useAuth'
import type { ApiError } from '@/core/api/request'
import LoginPage from '@/lawyer/pages/LoginPage.vue'
import WorkbenchPage from '@/lawyer/pages/WorkbenchPage.vue'
import NavPage from '@/lawyer/pages/NavPage.vue'
import ArticlePage from '@/lawyer/pages/ArticlePage.vue'
import CasesPage from '@/lawyer/pages/CasesPage.vue'
import AuditPage from '@/lawyer/pages/AuditPage.vue'

// 恰六条（设计 §四 律师侧路由表）：/login 不设 requiresAuth——否则登录页自己
// 被守卫挡住，未登录用户永远进不来（登录页是唯一免鉴权页）
export const routes: RouteRecordRaw[] = [
  { path: '/login', name: 'login', component: LoginPage },
  { path: '/', name: 'workbench', component: WorkbenchPage, meta: { requiresAuth: true } },
  { path: '/nav', name: 'nav', component: NavPage, meta: { requiresAuth: true } },
  { path: '/law/:lawId/articles/:no', name: 'article', component: ArticlePage, meta: { requiresAuth: true } },
  { path: '/cases', name: 'cases', component: CasesPage, meta: { requiresAuth: true } },
  { path: '/audit', name: 'audit', component: AuditPage, meta: { requiresAuth: true } },
]

// 工厂而不是「实例 + 事后 installGuards」：history 是唯一的注入点，守卫在
// 工厂内部装上——测试注入 memory history 拿到的就是**生产同一份**守卫，
// 不存在「实现忘了装守卫而测试自抄一份照样绿」的缝（brief 明说防的就是这个）
export function createLawyerRouter(history: RouterHistory): Router {
  const router = createRouter({ history, routes })
  router.beforeEach((to) => {
    // returnTo 保留用户原本要去的页：登录成功后回跳，而不是一律落到工作台
    if (to.meta.requiresAuth && session.token === null) {
      return { path: '/login', query: { returnTo: to.fullPath } }
    }
    return true
  })
  return router
}

// 专用路由 404 ⇒ 会话失效（④a 的 404 语义，归型在 request.ts：on404:'auth-expired'）。
// 页面 catch 里调用：true = 已处理（清 token + 跳登录，页面不要再渲染 ErrorPanel）。
// 越权判断收在这一处——每个页面各写一遍 if (kind === 'auth-expired') 时，
// 漏一处的表现是那页把「登录过期」显示成普通错误文案，没有红灯
export function handleAuthExpired(err: ApiError, router: Router): boolean {
  if (err.kind !== 'auth-expired') return false
  logout()
  void router.push('/login')
  return true
}

// 实例给 main.ts 挂载；生产 history 与测试注入的 memory history 各建各的。
// 必须用 hash 而不是 web history（终审 C1）：律师入口是文件级入口 /lawyer.html，
// install 期 vue-router 会 push(routerHistory.location)，web history 下那是
// '/lawyer.html'——路由表里没有这条，matched=0，router-view 渲染空、守卫也不参与，
// 浏览器打开律师入口就是白屏。hash 模式只吃 # 之后的部分，文件路径不进路由解析，
// 初始 hash 为空即落 '/'，由守卫带回 /login。替代方案 createWebHistory('/lawyer.html')
// 得到 /lawyer.html/login，刷新/深链需要服务器把该路径 rewrite 回 lawyer.html
// （dev 与生产目前都没有这条 rewrite，落点是公众 index.html），故不取。
// 机械偏离：律师侧 URL 变为 /lawyer.html#/login 形态（公开面记录在案）。
export const router = createLawyerRouter(createWebHashHistory())
