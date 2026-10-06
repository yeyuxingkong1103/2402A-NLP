<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'

type Source = { chunk_id: string; document_id?: string; title: string; page_start: number; page_end: number; score: number; rerank_score?: number; text: string }
type Message = { role: 'user' | 'assistant'; content: string; sources?: Source[]; safetyIntervention?: boolean }
type Conversation = { id: string; sessionId: string | null; title: string; roleId: string; roleName: string; messages: Message[] }
type User = { user_id: string; username: string; display_name: string }
type AIRole = { role_id: string; name: string; description: string; is_active: boolean }
type HistoryPayload = { conversations: Array<{ session_id: string; title: string; role_id: string; role_name: string; messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string; sources?: Source[]; safety_intervention?: boolean }> }> }
type ChatStreamEvent = { event: string; data: Record<string, any> }

const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'
const tokenKey = 'mentalheal_access_token'
const sessionId = ref<string | null>(null)
const activeId = ref(makeId())
const activeRoleId = ref('mental-health')
const input = ref('')
const loading = ref(false)
const progress = ref(0)
const progressStage = ref('准备处理问题')
const streamingIndex = ref<number | null>(null)
const progressTimer = ref<ReturnType<typeof setInterval> | null>(null)
const error = ref('')
const collapsed = ref(false)
const scrollArea = ref<HTMLElement | null>(null)
const messages = ref<Message[]>([])
const conversations = ref<Conversation[]>([])
const roles = ref<AIRole[]>([])
const currentUser = ref<User | null>(null)
const accessToken = ref(localStorage.getItem(tokenKey) || '')
const authMode = ref<'login' | 'register'>('login')
const authUsername = ref('')
const authPassword = ref('')
const authDisplayName = ref('')
const authError = ref('')
const authLoading = ref(false)
const editingId = ref<string | null>(null)
const editingTitle = ref('')

const hasMessages = computed(() => messages.value.some((message) => message.role === 'user'))
const sources = computed(() => {
  for (let index = messages.value.length - 1; index >= 0; index -= 1) {
    if (messages.value[index].sources?.length) return messages.value[index].sources || []
  }
  return []
})
const activeRole = computed(() => roles.value.find((role) => role.role_id === activeRoleId.value) || roles.value[0])
const suggestions = ['最近压力很大，怎么缓解？', '怎样改善睡眠质量？', '如何陪伴情绪低落的人？']

function makeId() { return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}` }
function copyMessages(items: Message[]) { return items.map((item) => ({ ...item, sources: item.sources ? [...item.sources] : undefined })) }
function authHeaders(json = false): HeadersInit {
  return {
    ...(json ? { 'Content-Type': 'application/json' } : {}),
    ...(accessToken.value ? { Authorization: `Bearer ${accessToken.value}` } : {}),
  }
}
function clearAuth() {
  accessToken.value = ''
  currentUser.value = null
  localStorage.removeItem(tokenKey)
  conversations.value = []
  messages.value = []
  sessionId.value = null
  activeId.value = makeId()
}
async function apiRequest(path: string, init: RequestInit = {}) {
  const response = await fetch(`${apiBase}${path}`, {
    ...init,
    headers: { ...authHeaders(Boolean(init.body)), ...(init.headers || {}) },
  })
  if (response.status === 401) {
    clearAuth()
    throw new Error('登录已失效，请重新登录')
  }
  if (!response.ok) {
    let detail = '请求失败'
    try { detail = String((await response.json()).detail || detail) } catch { /* response may not be JSON */ }
    throw new Error(detail)
  }
  return response
}
function applyHistory(payload: HistoryPayload) {
  conversations.value = payload.conversations.map((conversation) => ({
    id: conversation.session_id,
    sessionId: conversation.session_id,
    title: conversation.title,
    roleId: conversation.role_id,
    roleName: conversation.role_name,
    messages: conversation.messages
      .filter((message) => message.role === 'user' || message.role === 'assistant')
      .map((message) => ({
        role: message.role as 'user' | 'assistant',
        content: message.content,
        sources: message.sources,
        safetyIntervention: message.safety_intervention,
      })),
  }))
}
async function loadHistory() {
  const response = await apiRequest('/history')
  applyHistory(await response.json() as HistoryPayload)
}
async function loadRoles() {
  const response = await apiRequest('/roles')
  const payload = await response.json() as { roles: AIRole[] }
  roles.value = payload.roles
  if (!roles.value.some((role) => role.role_id === activeRoleId.value)) activeRoleId.value = roles.value[0]?.role_id || 'mental-health'
}
async function loadUserData() {
  if (!accessToken.value) return
  try {
    const response = await apiRequest('/auth/me')
    currentUser.value = await response.json() as User
    await Promise.all([loadRoles(), loadHistory()])
  } catch (requestError) {
    authError.value = requestError instanceof Error ? requestError.message : '登录状态已失效'
  }
}
async function submitAuth() {
  authError.value = ''
  authLoading.value = true
  try {
    const path = authMode.value === 'login' ? '/auth/login' : '/auth/register'
    const body = authMode.value === 'login'
      ? { username: authUsername.value.trim(), password: authPassword.value }
      : { username: authUsername.value.trim(), password: authPassword.value, display_name: authDisplayName.value.trim() }
    const response = await fetch(`${apiBase}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    const payload = await response.json()
    if (!response.ok) throw new Error(String(payload.detail || '登录失败'))
    accessToken.value = payload.access_token
    localStorage.setItem(tokenKey, accessToken.value)
    currentUser.value = payload.user as User
    authPassword.value = ''
    await Promise.all([loadRoles(), loadHistory()])
  } catch (requestError) {
    authError.value = requestError instanceof Error ? requestError.message : '请求失败，请稍后重试'
  } finally {
    authLoading.value = false
  }
}
async function logout() {
  try { if (accessToken.value) await apiRequest('/auth/logout', { method: 'POST' }) } catch { /* local logout still clears the token */ }
  clearAuth()
}
function syncConversation() {
  if (!hasMessages.value) return
  const first = messages.value.find((message) => message.role === 'user')
  const role = activeRole.value
  const snapshot: Conversation = {
    id: activeId.value,
    sessionId: sessionId.value,
    title: (first?.content || '新的心理健康问答').slice(0, 27),
    roleId: role?.role_id || activeRoleId.value,
    roleName: role?.name || '心理健康助手',
    messages: copyMessages(messages.value),
  }
  const old = conversations.value.find((item) => item.id === activeId.value)
  if (old) Object.assign(old, snapshot)
  else conversations.value.unshift(snapshot)
}
function newConversation() {
  if (loading.value) return
  activeId.value = makeId(); sessionId.value = null; messages.value = []; input.value = ''; error.value = ''; editingId.value = null
}
function openConversation(conversation: Conversation) {
  if (loading.value) return
  activeId.value = conversation.id
  sessionId.value = conversation.sessionId
  activeRoleId.value = conversation.roleId
  messages.value = copyMessages(conversation.messages)
  input.value = ''; error.value = ''; editingId.value = null
  scrollToBottom()
}
function chooseSuggestion(value: string) { input.value = value }
function startRename(conversation: Conversation) {
  editingId.value = conversation.id
  editingTitle.value = conversation.title
}
function cancelRename() { editingId.value = null; editingTitle.value = '' }
async function saveRename(conversation: Conversation) {
  const title = editingTitle.value.trim()
  if (!title || !conversation.sessionId) return
  try {
    await apiRequest(`/history/${conversation.sessionId}`, { method: 'PATCH', body: JSON.stringify({ title }) })
    conversation.title = title
    cancelRename()
  } catch (requestError) {
    error.value = requestError instanceof Error ? requestError.message : '会话重命名失败'
  }
}
async function deleteConversation(conversation: Conversation) {
  if (!conversation.sessionId || !window.confirm(`确定删除“${conversation.title}”吗？`)) return
  try {
    await apiRequest(`/history/${conversation.sessionId}`, { method: 'DELETE' })
    conversations.value = conversations.value.filter((item) => item.id !== conversation.id)
    if (activeId.value === conversation.id) newConversation()
  } catch (requestError) {
    error.value = requestError instanceof Error ? requestError.message : '会话删除失败'
  }
}
function stopProgressTimer() {
  if (progressTimer.value) clearInterval(progressTimer.value)
  progressTimer.value = null
}
function moveProgress(target: number, stage: string) {
  progressStage.value = stage
  stopProgressTimer()
  progressTimer.value = setInterval(() => {
    if (progress.value >= target) { stopProgressTimer(); return }
    progress.value = Math.min(target, progress.value + 1)
  }, 28)
}
function parseStreamEvent(raw: string): ChatStreamEvent | null {
  const lines = raw.split('\n')
  const event = lines.find((line) => line.startsWith('event:'))?.slice(6).trim() || ''
  const dataLine = lines.find((line) => line.startsWith('data:'))?.slice(5).trim()
  if (!dataLine) return null
  try { return { event, data: JSON.parse(dataLine) as Record<string, any> } } catch { return null }
}
async function readChatStream(response: Response, assistantIndex: number) {
  if (!response.body) throw new Error('浏览器不支持流式回答')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finished = false
  while (!finished) {
    const chunk = await reader.read()
    buffer += decoder.decode(chunk.value || new Uint8Array(), { stream: !chunk.done })
    const events = buffer.split('\n\n')
    buffer = events.pop() || ''
    for (const raw of events) {
      const parsed = parseStreamEvent(raw)
      if (!parsed) continue
      const { event, data } = parsed
      if (event === 'session') {
        sessionId.value = String(data.session_id || sessionId.value || '')
        if (sessionId.value) activeId.value = sessionId.value
      }
      if (event === 'progress') moveProgress(Number(data.value) || progress.value, String(data.stage || '正在处理'))
      if (event === 'sources') {
        messages.value[assistantIndex].sources = (data.sources as Source[]) || []
        moveProgress(Number(data.value) || 45, String(data.stage || '正在重排资料'))
      }
      if (event === 'delta') {
        messages.value[assistantIndex].content += String(data.content || '')
        if (progress.value < 90 && progressStage.value !== '正在生成回答') moveProgress(90, '正在生成回答')
        await scrollToBottom()
      }
      if (event === 'done') {
        sessionId.value = String(data.session_id || sessionId.value || '')
        if (sessionId.value) activeId.value = sessionId.value
        messages.value[assistantIndex].safetyIntervention = Boolean(data.safety_intervention)
        progress.value = 100; progressStage.value = '回答完成'; stopProgressTimer(); finished = true
      }
      if (event === 'error') throw new Error(String(data.message || '在线问答暂时不可用，请稍后重试'))
    }
    if (chunk.done) break
  }
  if (!finished) throw new Error('回答流意外中断，请重试')
}
async function retrieve() {
  const query = input.value.trim()
  if (!query || loading.value || !currentUser.value) return
  error.value = ''; progress.value = 4; progressStage.value = '准备处理问题'
  messages.value.push({ role: 'user', content: query })
  const assistantIndex = messages.value.push({ role: 'assistant', content: '' }) - 1
  streamingIndex.value = assistantIndex; input.value = ''; loading.value = true
  moveProgress(12, '正在连接问答服务'); await scrollToBottom()
  try {
    const response = await fetch(`${apiBase}/chat/stream`, {
      method: 'POST', headers: { ...authHeaders(true), Accept: 'text/event-stream' },
      body: JSON.stringify({ message: query, top_k: 5, session_id: sessionId.value, role_id: activeRoleId.value }),
    })
    if (response.status === 401) { clearAuth(); throw new Error('登录已失效，请重新登录') }
    if (!response.ok) throw new Error('在线问答暂时不可用，请稍后重试')
    await readChatStream(response, assistantIndex)
    syncConversation()
  } catch (requestError) {
    messages.value.splice(assistantIndex, 1)
    error.value = requestError instanceof Error ? requestError.message : '请求失败，请稍后重试'
  } finally {
    loading.value = false; streamingIndex.value = null; stopProgressTimer(); await scrollToBottom()
  }
}
function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); retrieve() }
}
function scrollToBottom() { return nextTick(() => { if (scrollArea.value) scrollArea.value.scrollTop = scrollArea.value.scrollHeight }) }
function pages(source: Source) { return source.page_start === source.page_end ? `第 ${source.page_start} 页` : `第 ${source.page_start}–${source.page_end} 页` }
function score(value: number) { return Number.isFinite(value) ? value.toFixed(3) : '—' }
onMounted(loadUserData)
</script>

<template>
  <main class="shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark">♥</span><b>心安 AI</b><i></i><span>AI 问答</span></div>
      <div v-if="currentUser" class="account"><span class="online"><em></em>在线</span><span class="avatar">{{ currentUser.display_name.slice(0, 1) }}</span><span class="visitor">{{ currentUser.display_name }}</span><button class="logout" @click="logout">⇥ 退出</button></div>
      <span v-else class="account-hint">安全登录后使用对话</span>
    </header>

    <section v-if="!currentUser" class="auth-gate">
      <div class="auth-card">
        <div class="auth-icon">♥</div><small>MENTALHEAL AI</small><h1>{{ authMode === 'login' ? '欢迎回来' : '创建你的账号' }}</h1><p>登录后，你的对话历史会安全地保存并与其他用户隔离。</p>
        <form @submit.prevent="submitAuth">
          <label>用户名<input v-model="authUsername" autocomplete="username" placeholder="输入用户名" required></label>
          <label v-if="authMode === 'register'">显示名称<input v-model="authDisplayName" autocomplete="nickname" placeholder="输入显示名称" required></label>
          <label>密码<input v-model="authPassword" type="password" :autocomplete="authMode === 'login' ? 'current-password' : 'new-password'" placeholder="至少 8 位密码" required></label>
          <p v-if="authError" class="auth-error">{{ authError }}</p><button class="auth-submit" :disabled="authLoading">{{ authLoading ? '处理中…' : authMode === 'login' ? '登录' : '注册并登录' }}</button>
        </form>
        <button class="auth-switch" @click="authMode = authMode === 'login' ? 'register' : 'login'; authError = ''">{{ authMode === 'login' ? '还没有账号？立即注册' : '已有账号？返回登录' }}</button>
      </div>
    </section>

    <div v-else :class="['layout', { 'evidence-collapsed': collapsed }]">
      <aside class="panel history">
        <div class="heading"><div><small>CONVERSATION</small><h2>历史对话</h2></div><button class="add" aria-label="新建对话" @click="newConversation">＋</button></div>
        <div v-if="conversations.length" class="conversation-list">
          <div v-for="conversation in conversations" :key="conversation.id" :class="['conversation-row', { selected: conversation.id === activeId }]">
            <button class="conversation" @click="openConversation(conversation)"><span>◌</span><strong>{{ conversation.title }}</strong><small>{{ conversation.roleName }}</small></button>
            <div v-if="editingId === conversation.id" class="rename-box"><input v-model="editingTitle" maxlength="120" @keyup.enter="saveRename(conversation)" @keyup.esc="cancelRename"><button @click="saveRename(conversation)">保存</button><button @click="cancelRename">取消</button></div>
            <div v-else class="conversation-actions"><button title="重命名" @click.stop="startRename(conversation)">✎</button><button title="删除" @click.stop="deleteConversation(conversation)">×</button></div>
          </div>
        </div>
        <div v-else class="empty history-empty"><span class="empty-icon">▱</span><b>暂无历史对话</b><p>发送第一个问题后，会在这里保存会话入口。</p></div>
        <footer class="privacy">● 当前账号的历史已隔离保存</footer>
      </aside>

      <section class="panel chat">
        <div class="role-head"><div class="role"><span class="role-icon">♥</span><div><small>ACTIVE ROLE</small><h2>{{ activeRole?.name || '心理健康助手' }}</h2><p>{{ activeRole?.description || '通用心理健康科普与陪伴' }}</p></div></div><div class="role-tools"><select v-model="activeRoleId" class="role-select" :disabled="loading"><option v-for="role in roles" :key="role.role_id" :value="role.role_id">{{ role.name }}</option></select><span class="online"><em></em>在线</span></div></div>
        <div class="safety"><span>♢</span><p><b>温馨提示：</b>本系统提供心理健康科普与陪伴式问答，不能替代医生诊断、心理治疗或紧急救助；如有自伤、他伤或急性危机，请立即联系身边可信任的人或当地急救服务。</p></div>
        <div ref="scrollArea" class="messages">
          <div v-if="!hasMessages" class="welcome"><div class="welcome-card"><span>♥</span><div><b>温和、稳妥，适合大多数心理健康问题</b><p>当前角色：{{ activeRole?.name || '心理健康助手' }}</p></div></div><div class="empty question-empty"><span class="empty-icon">▱</span><b>从一个轻松的问题开始</b><p>可以询问压力管理、心理急救、睡眠困扰或青少年心理健康等知识。</p><div class="suggestions"><button v-for="suggestion in suggestions" :key="suggestion" @click="chooseSuggestion(suggestion)">{{ suggestion }}</button></div></div></div>
          <div v-else class="message-list"><article v-for="(message, index) in messages" :key="`${index}-${message.role}`" :class="['message', message.role]"><span v-if="message.role === 'assistant'" class="message-avatar">♥</span><div class="message-body"><small>{{ message.role === 'assistant' ? (activeRole?.name || 'AI 助手') : '你' }}</small><div :class="['bubble', { warning: message.safetyIntervention }]" :aria-busy="loading && index === streamingIndex"><p v-if="message.content">{{ message.content }}</p><template v-if="loading && index === streamingIndex"><div class="progress-info"><span>{{ progressStage }}</span><b>{{ progress }}%</b></div><div class="progress-track" role="progressbar" :aria-valuenow="progress" aria-valuemin="0" aria-valuemax="100" aria-label="回答生成进度"><span :style="{ width: `${progress}%` }"></span></div></template><label v-if="message.sources?.length">已参考 {{ message.sources.length }} 条知识库内容</label></div></div></article></div>
        </div>
        <p v-if="error" class="error">{{ error }}</p>
        <form class="composer" @submit.prevent="retrieve"><textarea v-model="input" rows="1" placeholder="向心理健康助手提问..." @keydown="handleKeydown"></textarea><button :disabled="loading || !input.trim()">发送　➤</button></form><p class="hint">按 Enter 发送，Shift + Enter 换行</p>
      </section>

      <aside :class="['panel evidence', { collapsed }]"><div class="heading"><div><small>EVIDENCE</small><h2>参考来源</h2></div><button class="collapse" :aria-label="collapsed ? '展开参考来源' : '收起参考来源'" :title="collapsed ? '展开参考来源' : '收起参考来源'" @click.stop="collapsed = !collapsed"><span>{{ collapsed ? '展开' : '收起' }}</span><svg class="collapse-icon" :class="{ rotated: collapsed }" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5l7 7-7 7" /></svg></button></div><div v-if="!collapsed" class="evidence-body"><div v-if="sources.length" class="source-list"><article v-for="(source, index) in sources" :key="`${source.chunk_id}-${index}`"><div><b>0{{ index + 1 }}</b><span>相关度 {{ score(source.score) }}</span></div><h3>{{ source.title }}</h3><small>{{ pages(source) }}</small><p>{{ source.text }}</p></article></div><div v-else class="empty evidence-empty"><span class="empty-icon">▤</span><b>等待知识库引用</b><p>发送问题后，这里会展示检索到的文件、页码和相关度。</p></div></div></aside>
    </div>
  </main>
</template>

<style>
:root{color:#27352f;background:#f5f7f5;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;font-synthesis:none}*{box-sizing:border-box}html,body,#app{height:100%}body{margin:0;min-width:320px;overflow:hidden;background:#f5f7f5}button,textarea,input,select{font:inherit}button{cursor:pointer}.shell{height:100dvh;display:flex;flex-direction:column;overflow:hidden}.topbar{height:72px;flex:0 0 72px;display:flex;align-items:center;justify-content:space-between;padding:0 34px;background:#fff;border-bottom:1px solid #e5ece8}.brand,.account,.online,.role,.role-tools{display:flex;align-items:center}.brand{gap:13px}.brand-mark{width:35px;height:35px;display:grid;place-items:center;color:#198364;background:#e3f3ea;border-radius:50%;font-size:20px}.brand b{color:#1c2b25;font-size:19px}.brand i{width:1px;height:22px;margin:0 5px;background:#dce5df}.brand>span:last-child,.account,.account-hint{color:#6d7b74;font-size:13px}.account{gap:15px}.account button{border:0;background:transparent;color:#78847e}.memory-toggle{color:#238b6b!important}.memory-modal{position:fixed;z-index:10;inset:0;display:grid;place-items:center;padding:20px;background:#20372c55}.memory-card{width:min(520px,100%);max-height:min(720px,calc(100dvh - 40px));overflow-y:auto;padding:24px;background:#fff;border:1px solid #e2eae5;border-radius:14px;box-shadow:0 18px 60px #20372c33}.memory-card-header{display:flex;align-items:flex-start;justify-content:space-between}.memory-card-header small{color:#a0aaa5;font-size:10px;font-weight:700;letter-spacing:.15em}.memory-card-header h2{margin:6px 0 0;color:#28372f;font-size:18px}.memory-close{width:28px;height:28px;color:#78847e;border:0;background:#f1f8f4;border-radius:7px;font-size:20px}.memory-notice{margin:15px 0;color:#718078;background:#f5faf7;border:1px solid #e2f0e7;border-radius:7px;padding:10px 12px;font-size:11px;line-height:1.7}.memory-form{display:grid;gap:11px}.memory-form label{display:grid;gap:6px;color:#607169;font-size:11px;font-weight:600}.memory-form textarea,.memory-form select{width:100%;padding:9px 10px;color:#405048;background:#fbfcfb;border:1px solid #dfe9e3;border-radius:7px;outline:0;font-size:12px;line-height:1.6}.memory-form textarea:focus,.memory-form select:focus{border-color:#6db094;box-shadow:0 0 0 3px #e6f4ed}.memory-confirm{display:flex!important;grid-template-columns:16px 1fr;align-items:center;display:flex!important;gap:7px!important;font-weight:500!important}.memory-confirm input{accent-color:#238b6b}.memory-list{margin-top:22px}.memory-list-title{display:flex;justify-content:space-between;padding-bottom:8px;border-bottom:1px solid #edf1ee;color:#52665b;font-size:12px}.memory-list-title span{color:#9aaa9f;font-size:10px}.memory-row{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;padding:11px 0;border-bottom:1px solid #edf1ee}.memory-row p{margin:0;color:#607169;font-size:11px;line-height:1.7;white-space:pre-wrap}.memory-row button{flex:0 0 auto;padding:4px 7px;color:#b34b42;background:#fff;border:1px solid #f0dcd8;border-radius:5px;font-size:10px}.memory-empty{margin:15px 0;color:#9aa69f;font-size:11px}.account{gap:15px}.account button{border:0;background:transparent;color:#78847e}.online{gap:6px;color:#32836a}.online em{width:7px;height:7px;background:#41b487;border-radius:50%;box-shadow:0 0 0 3px #e3f5eb}.avatar{width:30px;height:30px;display:grid;place-items:center;color:#497966;background:#dcefe6;border-radius:50%;font-weight:700}.visitor{margin-left:-8px;color:#46554e}.auth-gate{flex:1;display:grid;place-items:center;padding:30px}.auth-card{width:min(420px,100%);padding:38px 42px;text-align:center;background:#fff;border:1px solid #e2eae5;border-radius:16px;box-shadow:0 12px 36px #2c4e3e12}.auth-icon{width:52px;height:52px;display:grid;place-items:center;margin:0 auto 15px;color:#198364;background:#e3f3ea;border-radius:16px;font-size:27px}.auth-card>small{color:#9aaa9f;font-size:10px;letter-spacing:.18em}.auth-card h1{margin:10px 0 8px;color:#28372f;font-size:24px}.auth-card>p{margin:0 0 24px;color:#86958c;font-size:12px;line-height:1.7}.auth-card form{display:grid;gap:14px;text-align:left}.auth-card label{display:grid;gap:6px;color:#607169;font-size:11px;font-weight:600}.auth-card input,.rename-box input{width:100%;padding:10px 11px;color:#405048;background:#fbfcfb;border:1px solid #dfe9e3;border-radius:7px;outline:0;font-size:12px}.auth-card input:focus,.rename-box input:focus{border-color:#6db094;box-shadow:0 0 0 3px #e6f4ed}.auth-submit{padding:11px;color:#fff;background:#238b6b;border:0;border-radius:7px;font-size:12px;font-weight:700}.auth-submit:disabled{opacity:.55}.auth-error{margin:0;color:#b34b42;font-size:11px}.auth-switch{margin-top:19px;color:#238b6b;border:0;background:transparent;font-size:11px}.layout{width:min(1500px,calc(100% - 48px));height:calc(100dvh - 104px);min-height:0;max-height:calc(100dvh - 104px);display:grid;grid-template-columns:245px minmax(420px,1fr) 285px;gap:16px;margin:16px auto;overflow:hidden}.panel{height:100%;min-height:0;background:#fff;border:1px solid #e2eae5;border-radius:12px;box-shadow:0 4px 18px #2c4e3e09}.history,.evidence{display:flex;flex-direction:column;padding:25px 19px 18px}.heading,.role-head{display:flex;align-items:flex-start;justify-content:space-between}.heading small,.role small{display:block;color:#a0aaa5;font-size:10px;font-weight:700;letter-spacing:.15em}.heading h2,.role h2{margin:6px 0 0;color:#28372f;font-size:17px}.add{width:29px;height:29px;color:#fff;border:0;border-radius:8px;background:#238b6b;font-size:21px;line-height:20px}.conversation-list{display:grid;gap:7px;margin-top:27px;overflow-y:auto}.conversation-row{position:relative;display:flex;align-items:center;border:1px solid transparent;border-radius:9px}.conversation-row.selected{background:#eff8f3;border-color:#dceee4}.conversation-row:hover{background:#f7fbf8}.conversation{display:grid;grid-template-columns:18px minmax(0,1fr);gap:4px 6px;flex:1;padding:10px 3px 10px 9px;color:#66736d;text-align:left;border:0;background:transparent}.conversation>span{grid-row:span 2;color:#5f9b82;font-size:21px}.conversation strong{overflow:hidden;font-size:12px;font-weight:600;text-overflow:ellipsis;white-space:nowrap}.conversation small{overflow:hidden;color:#9aaa9f;font-size:9px;text-overflow:ellipsis;white-space:nowrap}.conversation-actions{display:flex;padding-right:5px}.conversation-actions button,.rename-box button{padding:4px;color:#8da098;border:0;background:transparent;font-size:11px}.conversation-actions button:hover,.rename-box button:hover{color:#238b6b}.rename-box{position:absolute;z-index:2;inset:3px 3px 3px 3px;display:flex;align-items:center;gap:3px;padding:3px;background:#fff;border-radius:6px}.rename-box input{min-width:0;padding:6px;font-size:10px}.rename-box button{padding:3px;font-size:9px}.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;color:#9aa69f}.history-empty{flex:1;min-height:300px;padding:20px 12px}.empty-icon{width:56px;height:56px;display:grid;place-items:center;margin-bottom:16px;color:#9ab5a8;background:#f0f7f3;border-radius:50%;font-size:28px}.empty b{color:#718078;font-size:13px}.empty p{max-width:190px;margin:8px 0 0;color:#a0aaa5;font-size:11px;line-height:1.7}.privacy{display:flex;gap:7px;margin-top:auto;padding:15px 4px 0;color:#a3ada7;border-top:1px solid #edf1ee;font-size:10px}.chat{min-width:0;display:flex;flex-direction:column;overflow:hidden}.role-head{min-height:86px;padding:21px 27px 16px;border-bottom:1px solid #edf1ee}.role{gap:12px}.role-icon{width:42px;height:42px;display:grid;place-items:center;color:#198364;background:#e4f4eb;border-radius:11px;font-size:25px}.role h2{margin:3px 0 2px;font-size:15px}.role p{margin:0;color:#9aa59f;font-size:11px}.role-tools{gap:17px;padding-top:7px}.role-select{min-width:130px;padding:8px 10px;color:#66736d;border:1px solid #e2eae5;border-radius:7px;background:#fff;font-size:11px}.role-tools .online{font-size:11px}.safety{display:flex;gap:9px;margin:18px 27px 4px;padding:11px 13px;color:#8b6a29;background:#fff9e8;border:1px solid #f5e8bd;border-radius:7px}.safety>span{font-size:18px}.safety p{margin:0;font-size:10px;line-height:1.75}.messages{min-height:0;flex:1;overflow-y:auto;padding:17px 27px 4px}.welcome{min-height:100%;display:flex;flex-direction:column}.welcome-card{display:flex;align-items:center;gap:14px;padding:16px 19px;background:#f1f8f4;border:1px solid #e2f0e7;border-radius:9px}.welcome-card>span{width:43px;height:43px;display:grid;place-items:center;color:#4ca581;background:#dff1e7;border-radius:50%;font-size:25px}.welcome-card b{color:#355247;font-size:14px}.welcome-card p{margin:5px 0 0;color:#82978d;font-size:11px}.question-empty{flex:1;padding:32px 16px 50px}.question-empty p{max-width:330px}.suggestions{display:flex;flex-wrap:wrap;justify-content:center;gap:7px;margin-top:20px}.suggestions button{padding:7px 10px;color:#5c8e77;background:#f7fbf8;border:1px solid #dcece3;border-radius:15px;font-size:10px}.message-list{display:grid;gap:18px;padding:8px 0 20px}.message{display:flex;gap:10px;align-items:flex-start}.message.user{justify-content:flex-end}.message-body{max-width:min(78%,680px)}.message.user .message-body{display:flex;flex-direction:column;align-items:flex-end}.message-avatar{width:28px;height:28px;display:grid;flex:0 0 auto;place-items:center;color:#4a9b7c;background:#e5f4eb;border-radius:8px;font-size:17px}.message-body>small{display:block;margin:1px 0 5px;color:#9aa79f;font-size:10px}.bubble{padding:11px 14px;color:#4e6057;background:#f1f8f4;border:1px solid #e2f0e7;border-radius:4px 12px 12px 12px;font-size:12px;line-height:1.8;white-space:pre-wrap}.message.user .bubble{color:#fff;background:#278e6c;border-color:#278e6c;border-radius:12px 4px 12px 12px}.bubble.warning{color:#715d35;background:#fff9e8;border-color:#f4e7bd}.bubble p{margin:0}.bubble label{display:block;margin-top:7px;color:#76a28e;font-size:10px}.progress-info{display:flex;justify-content:space-between;gap:20px;margin-top:10px;color:#78a08e;font-size:10px}.progress-info b{color:#238b6b;font-weight:700}.progress-track{width:190px;height:5px;overflow:hidden;margin-top:5px;background:#dceee5;border-radius:99px}.progress-track span{display:block;height:100%;background:#238b6b;border-radius:99px;transition:width .18s ease-out}.error{margin:0 27px 8px;color:#b34b42;font-size:11px}.composer{display:flex;align-items:flex-end;gap:9px;margin:0 27px;padding:10px 10px 10px 13px;background:#fbfcfb;border:1px solid #dfe9e3;border-radius:9px}.composer textarea{min-height:27px;max-height:100px;flex:1;resize:vertical;padding:4px 0;color:#405048;border:0;outline:0;background:transparent;font-size:12px;line-height:1.6}.composer textarea::placeholder{color:#a6b0aa}.composer button{padding:8px 12px;color:#fff;background:#238b6b;border:0;border-radius:7px;font-size:11px;font-weight:600}.composer button:disabled{cursor:not-allowed;opacity:.45}.hint{margin:6px 28px 13px;color:#aab3ae;font-size:9px;text-align:right}.evidence-body{min-height:0;flex:1;overflow-y:auto}.source-list{display:grid;gap:10px;padding-top:15px}.source-list article{padding:12px;background:#fbfcfb;border:1px solid #e5eee8;border-radius:8px}.source-list article>div{display:flex;justify-content:space-between;margin-bottom:9px}.source-list article>div b{color:#9bb6a8;font-size:10px}.source-list article>div span{color:#4b9a78;font-size:9px}.source-list h3{margin:0 0 5px;color:#53645b;font-size:11px;line-height:1.5}.source-list article>small{color:#92a29a;font-size:9px}.source-list p{display:-webkit-box;overflow:hidden;margin:8px 0 0;color:#8b9891;font-size:10px;line-height:1.65;-webkit-box-orient:vertical;-webkit-line-clamp:5}.evidence-empty{min-height:280px;padding:20px 12px}.collapse{display:inline-flex;align-items:center;gap:5px;padding:6px 8px;color:#568c73;border:1px solid #dceee4;background:#f5faf7;border-radius:7px;font-size:10px;font-weight:600;transition:color .18s ease,background .18s ease}.collapse:hover{color:#238b6b;background:#e8f6ee}.collapse-icon{width:15px;height:15px;fill:none;stroke:currentColor;stroke-linecap:round;stroke-linejoin:round;stroke-width:2;transition:transform .2s ease}.collapse-icon.rotated{transform:rotate(180deg)}.collapsed{min-width:0;padding-left:10px;padding-right:10px}.collapsed .heading{align-items:center}.collapsed .heading>div{display:none}.collapsed .collapse{writing-mode:vertical-rl}.layout.evidence-collapsed{grid-template-columns:245px minmax(420px,1fr) 42px}.evidence-collapsed .evidence-body{display:none}@media(max-width:1180px){.layout{grid-template-columns:215px minmax(390px,1fr) 245px}.topbar{padding:0 24px}.visitor{display:none}}@media(max-width:920px){.layout{grid-template-columns:205px minmax(0,1fr)}.evidence{display:none}}@media(max-width:680px){.topbar{height:62px;flex-basis:62px;padding:0 15px}.brand{gap:8px}.brand-mark{width:31px;height:31px}.brand i,.brand>span:last-child,.logout,.account-hint{display:none}.layout{width:100%;height:calc(100dvh - 62px);min-height:0;max-height:none;display:block;margin:0}.history{display:none}.chat,.panel{height:100%;min-height:0;border:0;border-radius:0}.role-head{padding:16px}.role-tools{gap:0}.role-select{min-width:112px}.safety{margin:13px 16px 3px}.messages{padding:14px 16px 4px}.welcome-card{padding:14px}.welcome-card b{font-size:12px}.welcome-card p{font-size:10px}.message-body{max-width:84%}.composer{margin:0 16px}.hint{margin-right:17px;margin-left:17px}.auth-gate{padding:16px}.auth-card{padding:30px 24px}}
.admin-page{width:min(1180px,calc(100% - 48px));height:calc(100dvh - 104px);min-height:0;margin:16px auto;overflow-y:auto;display:grid;align-content:start;gap:14px}.admin-card{padding:22px 24px;background:#fff;border:1px solid #e2eae5;border-radius:12px;box-shadow:0 4px 18px #2c4e3e09}.admin-header{display:flex;align-items:flex-start;justify-content:space-between}.admin-header small{color:#a0aaa5;font-size:10px;font-weight:700;letter-spacing:.15em}.admin-header h1{margin:7px 0 5px;color:#28372f;font-size:22px}.admin-header p,.admin-note{margin:0;color:#8b9891;font-size:11px;line-height:1.7}.admin-refresh,.admin-actions button{padding:8px 11px;color:#4c806b;background:#f1f8f4;border:1px solid #dceee4;border-radius:7px;font-size:10px}.admin-refresh:disabled{opacity:.5}.admin-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.admin-card-title{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}.admin-card-title h2{margin:0;color:#40564b;font-size:14px}.admin-card-title span{color:#9aaa9f;font-size:10px}.upload-row{display:flex;align-items:center;gap:10px}.upload-row input{min-width:0;flex:1;color:#718078;font-size:10px}.admin-table{display:grid;border-top:1px solid #edf1ee}.admin-row{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:12px 0;border-bottom:1px solid #edf1ee}.admin-row>div:first-child{min-width:0;display:grid;gap:4px}.admin-row b{overflow:hidden;color:#52665b;font-size:11px;text-overflow:ellipsis;white-space:nowrap}.admin-row small{overflow:hidden;color:#9aa79f;font-size:9px;text-overflow:ellipsis;white-space:nowrap}.admin-actions{display:flex;flex:0 0 auto;gap:6px}.admin-actions button:hover{color:#238b6b;background:#e5f4eb}.job-status{flex:0 0 auto;color:#76877d;font-size:10px}.job-status.success{color:#268b69}.job-status.failed{color:#b34b42}.job-status.running{color:#a6772f}.metrics{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:5px;max-width:560px}.metrics span{padding:4px 6px;color:#4d866f;background:#f1f8f4;border-radius:5px;font-size:9px}.admin-error{margin:0}.admin-page .auth-submit{width:max-content;margin-top:13px}.admin-page .admin-note{margin-top:12px}@media(max-width:680px){.admin-page{width:100%;height:calc(100dvh - 62px);margin:0;padding:14px}.admin-header{padding:18px}.admin-header h1{font-size:18px}.admin-grid{grid-template-columns:1fr}.admin-card{padding:18px}.admin-row{align-items:flex-start;flex-direction:column}.admin-actions{width:100%}.metrics{justify-content:flex-start;max-width:none}.admin-refresh{font-size:9px}.account{gap:8px}.admin-toggle{display:none}}
</style>
