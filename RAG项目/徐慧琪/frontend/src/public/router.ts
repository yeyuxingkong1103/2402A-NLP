import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import AskPage from '@/public/pages/AskPage.vue'
import ArticlePage from '@/public/pages/ArticlePage.vue'
import NavPage from '@/public/pages/NavPage.vue'

// 具名导出 routes 是「公众入口只读」（FR-8.7）快照测试的判据对象
// （tests/public-router.spec.ts 钉「恰三条」）：任何新增路由——尤其上传/编辑/
// 删除/导出——都必须先过一次评审；写操作入口在这张字面量表上无处安放。
export const routes: RouteRecordRaw[] = [
  { path: '/', name: 'ask', component: AskPage },
  { path: '/nav', name: 'nav', component: NavPage },
  { path: '/law/:lawId/articles/:no', name: 'article', component: ArticlePage },
]

// 实例给 main.ts 挂载：history 模式与律师入口各建各的，两入口不共享运行时
export const router = createRouter({ history: createWebHistory(), routes })
