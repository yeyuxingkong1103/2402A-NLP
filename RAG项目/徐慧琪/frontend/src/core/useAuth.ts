// 律师侧 token 生命周期（设计 §四/§五）：登录写入、退出清除、注入给 request()。
// 本模块只被律师入口可达——公众入口零本地存储是 AC-19 的判据，把 localStorage
// 读写收在这一个文件里，公众侧漏用它的可能面就只剩「谁 import 了它」
// （source-scan.spec.ts 有一条扫公众侧 import 的护栏）。
import { reactive } from 'vue'
import { login as loginApi } from '@/core/api/lawyer'
import { setTokenProvider } from '@/core/api/request'

// 键名是跨任务字面量（测试与 hydration 都按它读，改一处必须两处同改）
const STORAGE_KEY = 'fl_lawyer_session'

export interface LawyerSession {
  token: string | null
  role: string | null
  teamId: string | null
  username: string | null
}

// reactive 而不是 ref：四个字段是一个整体（登出一起归零），页面直接读 session.role，
// 少一层 .value；模块级单例——token 在任何时刻只有一份真相
export const session = reactive<LawyerSession>({
  token: null, role: null, teamId: null, username: null,
})

// 刷新后从 localStorage 恢复：不回读的话「登录成功但刷新即掉线」——守卫看到
// token 为 null 会把已登录用户赶回登录页。坏 JSON（手工改过/半截写入）按未登录
// 处理而不是抛异常：抛在模块加载期会让整个律师入口白屏，比掉线更糟
function hydrate() {
  const raw = localStorage.getItem(STORAGE_KEY)
  if (!raw) return
  try {
    const saved = JSON.parse(raw) as Partial<LawyerSession>
    session.token = saved.token ?? null
    session.role = saved.role ?? null
    session.teamId = saved.teamId ?? null
    session.username = saved.username ?? null
  } catch {
    localStorage.removeItem(STORAGE_KEY)
  }
}
hydrate()

export async function login(username: string, password: string): Promise<void> {
  const res = await loginApi(username, password)
  session.token = res.access_token
  session.role = res.role
  session.teamId = res.team_id
  // username 不在 LoginResponse 里（后端不回显），由登录入参写入——顶部要显示
  // 「当前是谁」，这是唯一的信息来源
  session.username = username
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    token: session.token, role: session.role,
    teamId: session.teamId, username: session.username,
  }))
}

export function logout(): void {
  session.token = null
  session.role = null
  session.teamId = null
  session.username = null
  localStorage.removeItem(STORAGE_KEY)
}

// 注入给 request()：auth 请求头从 session 现读，登出后下一次请求自然不带 token。
// 在模块加载期接上——律师入口只有本模块碰 token，接线点就是这里，页面不必记得接
setTokenProvider(() => session.token)
