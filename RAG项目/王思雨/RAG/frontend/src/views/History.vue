<!-- 第 9 步：历史记录页面，按会话查看历史消息 -->
<template>
  <!-- 页面容器 -->
  <div class="page">
    <!-- 顶部：选择会话 -->
    <div class="head">
      <!-- 会话下拉框 -->
      <el-select v-model="current" placeholder="请选择会话" style="width: 320px"
                 @change="loadMessages">
        <!-- 逐条会话选项，显示标题与角色 -->
        <el-option v-for="c in conversations" :key="c.conversation_id"
                   :label="`#${c.conversation_id} ${c.title || '未命名'}（${c.role_name}）`"
                   :value="c.conversation_id" />
      </el-select>
      <!-- 手动刷新会话列表 -->
      <el-button @click="loadConversations">刷新列表</el-button>
    </div>

    <!-- 历史消息说明：来源不落库 -->
    <el-alert type="info" :closable="false" show-icon class="tip"
              title="历史消息只持久化问答正文；引用来源在对话页当场展示，未存入数据库。" />

    <!-- 消息列表 -->
    <div class="list">
      <!-- 无消息时占位 -->
      <el-empty v-if="messages.length === 0" description="该会话暂无消息" />
      <!-- 逐条渲染 -->
      <div v-for="(m, i) in messages" :key="i" class="row" :class="m.role">
        <!-- 气泡 -->
        <div class="bubble">
          <!-- 说话方标签 -->
          <span class="who">{{ m.role === 'user' ? '我' : '助手' }}</span>
          <!-- 正文 -->
          <div class="content">{{ m.content }}</div>
          <!-- 时间 -->
          <div class="time">{{ m.created_at }}</div>
          <!-- 引用来源：有数据才渲染（历史消息目前不带来源） -->
          <el-collapse v-if="m.sources && m.sources.length">
            <el-collapse-item :title="`引用来源（${m.sources.length}）`" name="s">
              <div v-for="(s, j) in m.sources" :key="j">
                <b>{{ s.source }}</b>
                <span class="score">相关度 {{ s.score }}</span>
              </div>
            </el-collapse-item>
          </el-collapse>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'                       // 导入响应式工具与挂载钩子
import { listConversations, listMessages } from '../api'   // 导入会话与消息接口

const conversations = ref([])                              // 当前用户的会话列表
const current = ref(null)                                  // 当前选中的会话编号
const messages = ref([])                                   // 当前会话的消息

// 拉取当前用户的会话列表
async function loadConversations() {
  const userId = localStorage.getItem('user_id')           // 当前用户编号
  const data = await listConversations(userId)             // 调接口
  conversations.value = data.conversations || []           // 存列表
  // 默认选中本地记录里的会话，没有就选第一条
  const saved = Number(localStorage.getItem('conversation_id'))
  current.value = conversations.value.some((c) => c.conversation_id === saved)
    ? saved : (conversations.value[0]?.conversation_id ?? null)
  if (current.value) await loadMessages()                  // 有选中会话就拉消息
}

// 拉取选中会话的消息
async function loadMessages() {
  if (!current.value) return                               // 没选会话直接返回
  const data = await listMessages(current.value)           // 调接口
  // 数据库只存了 role 与 content，sources 留空，保持结构一致便于扩展
  messages.value = (data.messages || []).map((m) => ({ ...m, sources: [] }))
}

onMounted(() => { loadConversations().catch(() => {}) })   // 挂载后加载，失败由拦截器提示
</script>

<style scoped>
/* 页面容器 */
.page {
  max-width: 900px;              /* 限制宽度 */
  margin: 0 auto;                /* 水平居中 */
}
/* 顶部工具栏 */
.head {
  display: flex;                 /* 横向排列 */
  gap: 12px;                     /* 元素间距 */
  align-items: center;           /* 垂直居中 */
}
/* 说明条 */
.tip {
  margin: 12px 0;                /* 上下留白 */
}
/* 消息列表 */
.list {
  padding: 12px;                 /* 内边距 */
  background: #fff;              /* 白底 */
  border: 1px solid #e4e7ed;     /* 边框 */
  border-radius: 4px;            /* 圆角 */
}
/* 单条消息一行 */
.row {
  display: flex;                 /* 弹性布局 */
  margin-bottom: 12px;           /* 消息间距 */
}
/* 用户消息靠右 */
.row.user {
  justify-content: flex-end;     /* 推到右侧 */
}
/* 助手消息靠左 */
.row.assistant {
  justify-content: flex-start;   /* 留在左侧 */
}
/* 气泡 */
.bubble {
  max-width: 80%;                /* 最宽 80% */
  padding: 8px 12px;             /* 内边距 */
  border-radius: 8px;            /* 圆角 */
  background: #f0f2f5;           /* 助手底色 */
}
/* 用户气泡底色区分 */
.row.user .bubble {
  background: #d9ecff;           /* 浅蓝 */
}
/* 说话方标签 */
.who {
  font-size: 12px;               /* 小字 */
  color: #909399;                /* 次要色 */
}
/* 正文换行处理 */
.content {
  white-space: pre-wrap;         /* 保留换行 */
  word-break: break-word;        /* 长词换行 */
  line-height: 1.6;              /* 行高 */
}
/* 时间戳 */
.time {
  margin-top: 4px;               /* 与正文留白 */
  font-size: 12px;               /* 小字 */
  color: #c0c4cc;                /* 更浅的颜色 */
}
/* 相关度分数 */
.score {
  margin-left: 8px;              /* 与文件名间距 */
  color: #909399;                /* 次要色 */
  font-size: 12px;               /* 小字 */
}
</style>
