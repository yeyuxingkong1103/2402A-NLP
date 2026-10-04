// 「无 token → 登录页」与「有 token → 放行」两向都要钉：只钉一向的话，
// 守卫变成「一律跳登录」也会全绿。用真实的 createLawyerRouter（只换 history），
// 判据对象必须是实现里的守卫本身。
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createMemoryHistory } from 'vue-router'
import { createLawyerRouter, router } from '@/lawyer/router'
import { login, logout } from '@/core/useAuth'

vi.mock('@/core/api/lawyer', () => ({
  login: vi.fn().mockResolvedValue({ access_token: 't', token_type: 'bearer', role: 'lawyer', team_id: 'team-a' }),
}))

function makeRouter() { return createLawyerRouter(createMemoryHistory()) }

beforeEach(() => { localStorage.clear(); logout() })

describe('律师侧守卫', () => {
  it('无 token 访问工作台 → 跳登录并带 returnTo', async () => {
    const r = makeRouter()
    await r.push('/')
    expect(r.currentRoute.value.path).toBe('/login')
    expect(r.currentRoute.value.query.returnTo).toBe('/')
  })

  it('有 token 访问工作台 → 放行（反向判据）', async () => {
    await login('u', 'p')
    const r = makeRouter()
    await r.push('/')
    expect(r.currentRoute.value.path).toBe('/')
  })
})

// 默认实例的 history 工厂（终审 C1）：律师入口是文件级 /lawyer.html，web history
// 下初始位置 '/lawyer.html' 无匹配路由 → matched=0 → 白屏（登录表单都不出现）。
// createHref 的形态只由 history 实现决定，是「生产实例到底是不是 hash」的直接
// 判别：把 createWebHashHistory 改回 createWebHistory 这条必红（期望 '#/login'
// 形态，web history 给 '/login'）。工厂签名不变，测试仍注入 memory history。
describe('律师侧默认实例的 history 形态', () => {
  it('默认实例是 hash history：createHref 产出 #/login 形态（改回 web history 必红）', () => {
    expect(router.options.history.createHref('/login')).toBe('#/login')
  })
})
