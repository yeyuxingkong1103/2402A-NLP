// 第 9 步：后端接口统一封装（axios + 手写 SSE 解析）

import axios from 'axios'          // 导入 axios 请求库
import { ElMessage } from 'element-plus'   // 导入消息提示组件，用于统一报错

const http = axios.create({        // 创建 axios 实例，统一配置
  baseURL: '/api',                 // 走 Vite 代理，实际转发到 http://127.0.0.1:8000
  timeout: 60000                   // 普通请求 60 秒超时（流式请求走 fetch，不受此限制）
})

http.interceptors.request.use((config) => {                 // 请求拦截器：自动带上用户标识
  const userId = localStorage.getItem('user_id')            // 从本地存储读取当前用户编号
  if (userId) {                                             // 已登录才带
    config.headers['X-User-Id'] = userId                    // 后端据此做多用户隔离
  }
  return config                                             // 放行请求
})

http.interceptors.response.use(
  (resp) => resp.data,                                      // 成功：直接返回业务数据，页面少写一层 .data
  (err) => {                                                // 失败：统一提示，页面不用各自处理
    const detail = err.response?.data?.detail || err.message || '请求失败'   // 优先用后端给的说明
    ElMessage.error(String(detail))                         // 弹一条错误提示
    return Promise.reject(err)                              // 继续抛出，页面可选地做收尾
  }
)

// ===================== 健康检查与用户 =====================

export const health = () => http.get('/health')                        // 健康检查：三项依赖状态
export const register = (username, password) =>                        // 注册
  http.post('/auth/register', { username, password })
export const login = (username, password) =>                           // 登录，返回 user_id 与 token
  http.post('/auth/login', { username, password })
export const getUser = (userId) => http.get(`/user/${userId}`)         // 查询用户信息

// ===================== 角色 =====================

export const listRoles = () => http.get('/roles')                      // 角色列表，返回 { roles: [...] }

// ===================== 会话与消息 =====================

export const createConversation = (userId, roleId, title = '新会话') =>   // 创建会话
  http.post('/conversation', { user_id: userId, role_id: roleId, title })
export const listConversations = (userId) =>                           // 某用户的会话列表
  http.get(`/conversation/${userId}`)
export const deleteConversation = (convId, userId) =>                  // 删除会话（带 user_id 做归属校验）
  http.delete(`/conversation/${convId}`, { params: { user_id: userId } })
export const listMessages = (convId, limit = 50) =>                    // 某会话的消息列表
  http.get(`/message/${convId}`, { params: { limit } })

// ===================== 问答 =====================

export const chatNonStream = (payload) => http.post('/chat', { ...payload, stream: false })   // 非流式问答

/**
 * 流式问答：用 fetch + ReadableStream 手写 SSE 解析。
 * EventSource 只支持 GET，本接口是 POST，所以必须手写。
 * @returns {AbortController} 调用方可 abort() 主动停止生成
 */
export function chatStream(payload, onChunk, onEnd, onError) {
  const ctrl = new AbortController()             // 用于中途停止流
  const userId = localStorage.getItem('user_id') // 当前用户编号

  fetch('/api/chat', {                           // 走 Vite 代理发请求
    method: 'POST',                              // 问答接口是 POST
    headers: { 'Content-Type': 'application/json', 'X-User-Id': userId || '' },   // 带用户标识
    body: JSON.stringify({ ...payload, stream: true }),   // 开启流式
    signal: ctrl.signal                          // 绑定中断信号
  }).then(async (resp) => {                      // 拿到响应
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)   // 非 200 直接报错
    const reader = resp.body.getReader()         // 取可读流读取器
    const decoder = new TextDecoder('utf-8')     // UTF-8 解码器，避免中文截断乱码
    let buffer = ''                              // 跨次读取的残留缓冲，保证帧完整
    for (;;) {                                   // 持续读取直到流结束
      const { done, value } = await reader.read()   // 读一块
      if (done) break                            // 流结束，退出循环
      buffer += decoder.decode(value, { stream: true })   // 追加解码结果（保留半个汉字）
      const frames = buffer.split('\n\n')        // 按 SSE 规范以空行切帧
      buffer = frames.pop()                      // 最后一段可能不完整，留到下一轮
      for (const frame of frames) {              // 逐帧处理
        if (!frame.startsWith('data: ')) continue   // 非数据帧跳过
        const text = frame.slice(6)              // 去掉 "data: " 前缀
        if (text === '[DONE]') { onEnd(null); continue }   // 结束标记：通知前端收尾
        try {                                    // 尝试按 JSON 解析
          const obj = JSON.parse(text)           // 解析成功说明是结束信息帧
          if (obj.error) onError(obj.error)      // 错误帧
          else onEnd(obj)                        // 带 sources / usage 的结束帧
        } catch {                                // 解析失败说明是正文片段
          onChunk(text)                          // 作为正文追加显示
        }
      }
    }
  }).catch((err) => {                            // 网络异常或主动中断
    if (err.name !== 'AbortError') onError(err.message)   // 主动停止不算错误
  })
  return ctrl                                    // 返回控制器，页面可据此实现“停止”
}

// ===================== 知识库、评测与缓存 =====================

export const uploadFile = (file, onProgress) => {   // 上传 PDF 并解析
  const form = new FormData()                       // 组装 multipart 表单
  form.append('file', file)                         // 字段名必须与后端 file 参数一致
  return http.post('/kb/upload', form, {            // 提交
    onUploadProgress: (e) => {                      // 上传进度回调
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))   // 换算百分比
    }
  })
}
export const rebuildKb = () => http.post('/kb/rebuild')      // 触发知识库重建（后台执行）
export const kbStatus = () => http.get('/kb/status')         // 知识库状态：集合名 / 条数 / 更新时间
export const evalResult = () => http.get('/eval/result')     // RAGAS 评测结果，无数据时返回空数组
export const clearCache = (userId) =>                        // 清理当前用户的短期记忆与答案缓存
  http.get('/cache/clear', { params: { user_id: userId } })
export const cacheStats = () => http.get('/cache/stats')     // 缓存统计：键总数、按用户分组、短期记忆用户数

export default http                                          // 导出实例，便于临时扩展
