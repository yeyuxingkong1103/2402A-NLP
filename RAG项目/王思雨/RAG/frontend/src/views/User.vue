<!-- 第 9 步：用户中心，展示账号信息与会话列表 -->
<template>
  <!-- 页面容器 -->
  <div class="page">
    <!-- 账号信息卡片 -->
    <el-card>
      <!-- 卡片标题 -->
      <template #header>我的账号</template>
      <!-- 用户信息描述列表 -->
      <el-descriptions :column="2" border>
        <!-- 用户编号 -->
        <el-descriptions-item label="用户编号">{{ userId }}</el-descriptions-item>
        <!-- 用户名 -->
        <el-descriptions-item label="用户名">{{ username }}</el-descriptions-item>
        <!-- 创建时间，接口拿不到时显示占位 -->
        <el-descriptions-item label="创建时间">{{ info.created_at || '-' }}</el-descriptions-item>
        <!-- 当前角色 -->
        <el-descriptions-item label="当前角色">{{ roleName || '未选择' }}</el-descriptions-item>
      </el-descriptions>
      <!-- 操作按钮区 -->
      <div class="actions">
        <!-- 重新选择角色 -->
        <el-button @click="$router.push('/role')">切换角色</el-button>
        <!-- 退出登录：清本地存储并回登录页 -->
        <el-button type="danger" @click="doLogout">退出登录</el-button>
      </div>
    </el-card>

    <!-- 会话列表卡片 -->
    <el-card class="gap">
      <!-- 卡片标题带刷新按钮 -->
      <template #header>
        <span>我的会话（{{ conversations.length }} 条）</span>
        <el-button link type="primary" class="refresh" @click="loadConversations">刷新</el-button>
      </template>
      <!-- 会话表格 -->
      <el-table :data="conversations" empty-text="暂无会话，去对话页发起一次提问吧">
        <!-- 会话编号 -->
        <el-table-column prop="conversation_id" label="编号" width="80" />
        <!-- 会话标题 -->
        <el-table-column prop="title" label="标题" />
        <!-- 使用的角色 -->
        <el-table-column prop="role_name" label="角色" width="140" />
        <!-- 创建时间 -->
        <el-table-column prop="created_at" label="创建时间" width="180" />
        <!-- 操作列 -->
        <el-table-column label="操作" width="160">
          <template #default="{ row }">
            <!-- 进入该会话继续对话 -->
            <el-button link type="primary" @click="openConversation(row)">打开</el-button>
            <!-- 删除该会话，带确认气泡 -->
            <el-popconfirm title="确认删除该会话及其消息？" @confirm="doDelete(row)">
              <template #reference>
                <el-button link type="danger">删除</el-button>
              </template>
            </el-popconfirm>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'          // 导入响应式工具与挂载钩子
import { useRouter } from 'vue-router'                  // 导入路由
import { ElMessage } from 'element-plus'                // 导入消息提示
import { deleteConversation, getUser, listConversations } from '../api'   // 导入用户与会话接口

const router = useRouter()                              // 路由实例
const userId = ref(localStorage.getItem('user_id') || '')   // 当前用户编号，来自本地存储
const username = ref(localStorage.getItem('username') || '')   // 当前用户名
const roleName = ref(localStorage.getItem('role_name') || '')   // 当前角色名
const info = reactive({ created_at: '' })               // 用户详细信息
const conversations = ref([])                           // 会话列表

// 拉取用户详情
async function loadUser() {
  const data = await getUser(userId.value)              // 调接口
  username.value = data.username                        // 以后端为准刷新用户名
  info.created_at = data.created_at                     // 记录创建时间
}

// 拉取会话列表
async function loadConversations() {
  const data = await listConversations(userId.value)    // 调接口
  conversations.value = data.conversations || []        // 存列表，缺失时给空数组
}

// 打开某个会话：把会话编号写入本地存储后跳到对话页
function openConversation(row) {
  localStorage.setItem('conversation_id', String(row.conversation_id))   // 对话页据此续聊
  router.push('/chat')                                  // 跳转
}

// 删除会话：成功后刷新列表
async function doDelete(row) {
  await deleteConversation(row.conversation_id, userId.value)   // 调接口，后端会校验归属
  ElMessage.success('已删除')                            // 提示
  if (localStorage.getItem('conversation_id') === String(row.conversation_id)) {   // 删的正是当前会话
    localStorage.removeItem('conversation_id')          // 一并清掉本地记录
  }
  await loadConversations()                             // 刷新列表
}

// 退出登录：清本地存储后回登录页
function doLogout() {
  localStorage.removeItem('user_id')                    // 清用户编号
  localStorage.removeItem('username')                   // 清用户名
  localStorage.removeItem('role_id')                    // 清角色编号
  localStorage.removeItem('role_name')                  // 清角色名
  localStorage.removeItem('conversation_id')            // 清会话编号
  ElMessage.success('已退出登录')                        // 提示
  router.push('/login')                                 // 回登录页
}

onMounted(() => {                                       // 页面挂载后加载数据
  loadUser().catch(() => {})                            // 用户详情失败不阻塞页面
  loadConversations().catch(() => {})                   // 会话列表失败同样不阻塞
})
</script>

<style scoped>
/* 页面容器 */
.page {
  max-width: 1000px;      /* 限制最大宽度，阅读更舒适 */
  margin: 0 auto;         /* 水平居中 */
}
/* 卡片之间的间距 */
.gap {
  margin-top: 16px;       /* 上间距 */
}
/* 标题栏里的刷新按钮靠右 */
.refresh {
  float: right;           /* 浮动到右侧 */
}
/* 按钮区上间距 */
.actions {
  margin-top: 16px;       /* 与描述列表拉开距离 */
}
</style>
