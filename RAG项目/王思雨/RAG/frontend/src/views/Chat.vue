<!-- 第 9 步：智能问答页面（流式输出 + 引用来源） -->
<template>
  <!-- 页面容器 -->
  <div class="page">
    <!-- 顶部：当前角色与长期记忆提示 -->
    <div class="head">
      <!-- 当前角色名 -->
      <span>当前角色：<b>{{ roleName }}</b></span>
      <!-- 会话编号，便于核对 -->
      <el-tag size="small" type="info">会话 {{ conversationId || '未创建' }}</el-tag>
      <!-- 换个角色入口 -->
      <el-button link type="primary" @click="$router.push('/role')">切换角色</el-button>
    </div>

    <!-- 消息列表区域 -->
    <div ref="listRef" class="list">
      <!-- 无消息时的占位 -->
      <el-empty v-if="messages.length === 0" description="输入问题开始提问，例如：绝缘油气相色谱分析步骤" />
      <!-- 逐条渲染消息 -->
      <div v-for="(m, i) in messages" :key="i" class="row" :class="m.role">
        <!-- 消息气泡 -->
        <div class="bubble">
          <!-- 正文内容，pre-wrap 保留换行 -->
          <div class="content">{{ m.content || '…' }}</div>
          <!-- 助手消息下方的附加信息 -->
          <div v-if="m.role === 'assistant'" class="meta">
            <!-- 本轮用到的长期记忆条数 -->
            <el-tag v-if="m.longMemory !== undefined" size="small" type="warning">
              长期记忆 {{ m.longMemory }} 条
            </el-tag>
            <!-- 引用来源，可折叠 -->
            <el-collapse v-if="m.sources && m.sources.length" class="sources">
              <el-collapse-item :title="`引用来源（${m.sources.length}）`" name="s">
                <!-- 逐条来源 -->
                <div v-for="(s, j) in m.sources" :key="j" class="source">
                  <b>{{ s.source }}</b>
                  <span class="score">相关度 {{ s.score }}</span>
                  <div class="summary">{{ s.summary }}</div>
                </div>
              </el-collapse-item>
            </el-collapse>
          </div>
        </div>
      </div>
    </div>

    <!-- 底部输入区 -->
    <div class="input-bar">
      <!-- 多行输入框，流式过程中禁用 -->
      <el-input v-model="draft" type="textarea" :rows="3" :disabled="streaming"
                placeholder="请输入电力维修相关问题，Ctrl+Enter 快速发送"
                @keydown.ctrl.enter.prevent="send" />
      <!-- 发送按钮：流式中变成停止 -->
      <div class="buttons">
        <el-button v-if="!streaming" type="primary" @click="send">发送</el-button>
        <el-button v-else type="danger" @click="stop">停止</el-button>
        <!-- 清空：丢弃当前上下文，另开一条新会话 -->
        <el-button :disabled="streaming" @click="clearAll">清空</el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { nextTick, onMounted, ref } from 'vue'                        // 导入响应式工具与挂载钩子
import { ElMessage } from 'element-plus'                              // 导入消息提示
import { chatStream, createConversation, listMessages } from '../api'   // 导入问答与会话接口

const roleName = ref(localStorage.getItem('role_name') || '未选角色')   // 当前角色名
const conversationId = ref(localStorage.getItem('conversation_id') || '')   // 当前会话编号
const messages = ref([])                                              // 消息列表：{role, content, sources, longMemory}
const draft = ref('')                                                 // 输入框内容
const streaming = ref(false)                                          // 是否正在流式接收
const listRef = ref(null)                                             // 消息列表 DOM，用于自动滚到底部
let ctrl = null                                                       // 当前流的 AbortController

// 滚动到列表底部，保证新内容可见
function scrollToBottom() {
  nextTick(() => {                                                    // 等 DOM 更新完再滚
    if (listRef.value) listRef.value.scrollTop = listRef.value.scrollHeight   // 直接置底
  })
}

// 确保存在会话：没有就新建一条并写入本地存储
async function ensureConversation() {
  if (conversationId.value) return conversationId.value               // 已有会话直接复用
  const userId = Number(localStorage.getItem('user_id'))               // 当前用户
  const roleId = Number(localStorage.getItem('role_id'))               // 当前角色
  const data = await createConversation(userId, roleId, '新会话')      // 建会话
  conversationId.value = String(data.conversation_id)                  // 记录编号
  localStorage.setItem('conversation_id', conversationId.value)        // 落本地存储，刷新不丢
  return conversationId.value                                          // 返回编号
}

// 发送问题：先入列表，再流式接收回答
async function send() {
  const text = draft.value.trim()                                      // 去掉首尾空白
  if (!text || streaming.value) return                                 // 空内容或正在生成时忽略
  try {
    await ensureConversation()                                         // 保证有会话
  } catch {
    return                                                             // 建会话失败，拦截器已提示
  }
  messages.value.push({ role: 'user', content: text, sources: [] })     // 用户消息入列表
  draft.value = ''                                                     // 清空输入框
  // 先占位一条空的助手消息，收到 chunk 后往里追加
  const idx = messages.value.push({ role: 'assistant', content: '', sources: [] }) - 1
  streaming.value = true                                               // 进入流式状态
  scrollToBottom()                                                     // 滚到底部

  ctrl = chatStream(                                                   // 发起流式问答
    { user_id: Number(localStorage.getItem('user_id')),                // 用户编号，多轮都要带
      role_id: Number(localStorage.getItem('role_id')),                // 角色编号
      conversation_id: Number(conversationId.value),                   // 会话编号，保证续在同一会话
      query: text },                                                   // 本轮问题
    (chunk) => {                                                       // 正文片段回调
      messages.value[idx].content += chunk                             // 追加到助手消息
      scrollToBottom()                                                 // 跟随滚动
    },
    (payload) => {                                                     // 结束回调：带 sources 的结束帧或 [DONE]
      if (payload) {                                                   // 结束信息帧
        messages.value[idx].sources = payload.sources || []            // 挂上引用来源
        messages.value[idx].longMemory = payload.long_memory_used ?? 0   // 挂上长期记忆条数
      }
      streaming.value = false                                          // 收尾，解除禁用
      scrollToBottom()                                                 // 最后滚一次
    },
    (err) => {                                                         // 错误回调
      messages.value[idx].content += `\n[出错] ${err}`                  // 把错误显示在气泡里
      streaming.value = false                                          // 收尾
    }
  )
}

// 停止生成：中断 fetch 流
function stop() {
  if (ctrl) ctrl.abort()                                               // 触发 AbortController
  streaming.value = false                                              // 立即恢复输入
  ElMessage.info('已停止生成')                                          // 提示
}

// 清空：丢弃当前上下文，另开一条新会话
async function clearAll() {
  localStorage.removeItem('conversation_id')                           // 清掉本地会话编号
  conversationId.value = ''                                            // 清掉页面内编号
  messages.value = []                                                  // 清空消息列表
  await ensureConversation()                                           // 立刻新建一条会话
  ElMessage.success('已开启新会话')                                      // 提示
}

onMounted(async () => {                                                // 页面挂载后拉历史
  if (!conversationId.value) return                                    // 没有会话就等首次提问时创建
  try {
    const data = await listMessages(conversationId.value)               // 拉该会话历史消息
    // 历史消息只存了正文，没有来源，来源仅本次会话内可见
    messages.value = (data.messages || []).map((m) => ({ role: m.role, content: m.content, sources: [] }))
    scrollToBottom()                                                   // 滚到底部
  } catch {
    /* 拉取失败时保持空列表，拦截器已提示 */
  }
})
</script>

<style scoped>
/* 页面容器：限制宽度并水平居中 */
.page { max-width: 900px; margin: 0 auto; }
/* 顶部信息条：角色名、会话号、切换入口一行排开 */
.head { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
/* 消息列表：固定高度、可滚动、白底带边框 */
.list { height: 55vh; overflow-y: auto; padding: 12px; background: #fff;
        border: 1px solid #e4e7ed; border-radius: 4px; }
/* 每条消息一行：用户靠右、助手靠左 */
.row { display: flex; margin-bottom: 12px; }
.row.user { justify-content: flex-end; }
.row.assistant { justify-content: flex-start; }
/* 气泡：助手浅灰底，用户浅蓝底 */
.bubble { max-width: 78%; padding: 8px 12px; border-radius: 8px; background: #f0f2f5; }
.row.user .bubble { background: #d9ecff; }
/* 正文：保留换行，长词强制换行 */
.content { white-space: pre-wrap; word-break: break-word; line-height: 1.6; }
/* 气泡内附加信息：长期记忆标签与来源折叠面板 */
.meta { margin-top: 8px; }
/* 来源条目：条目间用虚线分隔 */
.source { padding: 4px 0; border-bottom: 1px dashed #dcdfe6; }
/* 来源的相关度分数与摘要 */
.score { margin-left: 8px; color: #909399; font-size: 12px; }
.summary { color: #606266; font-size: 12px; }
/* 底部输入区与右侧按钮 */
.input-bar { margin-top: 12px; }
.buttons { margin-top: 8px; text-align: right; }
</style>
