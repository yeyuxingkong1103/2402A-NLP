// token 生命周期集中在这一个模块：登录写入、退出清除、注入给 request()。
// 判据要过真实 localStorage（jsdom 提供），而不是替身——存错键名是最常见的
// 「登录成功但刷新即掉线」的成因。
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { session, login, logout } from '@/core/useAuth'

vi.mock('@/core/api/lawyer', () => ({
  login: vi.fn().mockResolvedValue({ access_token: 'tok-9', token_type: 'bearer', role: 'partner', team_id: 'team-a' }),
}))

beforeEach(() => { localStorage.clear(); logout() })

describe('useAuth', () => {
  it('登录写入 localStorage 与 session', async () => {
    await login('u', 'p')
    expect(session.token).toBe('tok-9')
    expect(session.role).toBe('partner')
    expect(JSON.parse(localStorage.getItem('fl_lawyer_session')!).token).toBe('tok-9')
  })

  it('退出清除两处', async () => {
    await login('u', 'p')
    logout()
    expect(session.token).toBeNull()
    expect(localStorage.getItem('fl_lawyer_session')).toBeNull()
  })
})

// Task 6 首审 m1：既有两条只钉 token/role，teamId/username 的写入与刷新恢复
// 无判据——顶部身份会显示 null、hydrate 被删后旧套件全绿
describe('useAuth 刷新与完整写入', () => {
  it('登录写入完整 session（teamId/username）与 localStorage（顶部身份的数据源）', async () => {
    // Task 6 首审 m1 缺口：既有断言只钉 token/role，删掉 teamId/username 的写入
    // 全量测试仍绿——顶部「当前是谁、哪个团队」会显示 null，且刷新恢复也丢字段
    await login('u', 'p')
    expect(session.teamId).toBe('team-a')
    expect(session.username).toBe('u')
    const saved = JSON.parse(localStorage.getItem('fl_lawyer_session')!)
    expect(saved.teamId).toBe('team-a')
    expect(saved.username).toBe('u')
  })

  it('刷新后从 localStorage 恢复（hydrate：登录成功不能一刷新就掉线）', async () => {
    // Task 6 首审 m1 缺口：删掉 hydrate() 调用后全量 117 条仍绿。判据必须走
    // 「模块首次加载」这条路径——先手工预置存储，再 resetModules 后动态 import：
    // 静态 import 的模块已经加载过，不会为这条用例重跑 hydrate
    localStorage.setItem('fl_lawyer_session', JSON.stringify({
      token: 'tok-h', role: 'lawyer', teamId: 'team-h', username: 'u-h',
    }))
    vi.resetModules()
    const fresh = await import('@/core/useAuth')
    expect(fresh.session.token).toBe('tok-h')
    expect(fresh.session.role).toBe('lawyer')
    expect(fresh.session.teamId).toBe('team-h')
    expect(fresh.session.username).toBe('u-h')
  })
})
