<!-- 第 9 步：后台状态页面，全部只读 -->
<template>
  <div class="page">
    <!-- 顶部刷新 -->
    <div class="head">
      <h2>后台状态</h2>
      <el-button type="primary" :loading="loading" @click="loadAll">刷新</el-button>
    </div>

    <!-- 三项依赖服务状态 -->
    <el-card>
      <template #header>依赖服务状态（GET /health）</template>
      <el-descriptions :column="3" border>
        <el-descriptions-item label="Milvus">
          <el-tag :type="svc.milvus ? 'success' : 'danger'" size="small">
            {{ svc.milvus ? '可连' : '不可连' }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="MySQL">
          <el-tag :type="svc.mysql ? 'success' : 'danger'" size="small">
            {{ svc.mysql ? '可连' : '不可连' }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="Redis">
          <el-tag :type="svc.redis ? 'success' : 'danger'" size="small">
            {{ svc.redis ? '可连' : '不可连' }}
          </el-tag>
        </el-descriptions-item>
      </el-descriptions>
    </el-card>

    <!-- MySQL 表清单 -->
    <el-card class="gap">
      <template #header>MySQL 表清单（只读）</template>
      <el-table :data="tables" size="small">
        <!-- 表名 -->
        <el-table-column prop="name" label="表名" width="160" />
        <!-- 用途 -->
        <el-table-column prop="desc" label="用途" width="220" />
        <!-- 可读到的条数：现有接口能拿到的才显示数字 -->
        <el-table-column prop="count" label="条数" width="100" />
        <!-- 统计口径说明 -->
        <el-table-column prop="scope" label="统计口径" />
      </el-table>
    </el-card>

    <!-- Milvus 集合信息 -->
    <el-card class="gap">
      <template #header>Milvus Collection（只读）</template>
      <el-descriptions :column="1" border>
        <el-descriptions-item label="集合名">{{ kb.collection || '-' }}</el-descriptions-item>
        <el-descriptions-item label="向量条数">{{ kb.row_count }}</el-descriptions-item>
        <el-descriptions-item label="最后更新">{{ kb.last_update || '-' }}</el-descriptions-item>
      </el-descriptions>
    </el-card>

    <!-- Redis 缓存 -->
    <el-card class="gap">
      <template #header>Redis 缓存（只读）</template>
      <el-descriptions :column="1" border>
        <el-descriptions-item label="连通状态">
          <el-tag :type="svc.redis ? 'success' : 'danger'" size="small">
            {{ svc.redis ? '可连' : '不可连' }}
          </el-tag>
        </el-descriptions-item>
        <!-- 缓存键总数：来自 GET /cache/stats 的 total_keys -->
        <el-descriptions-item label="缓存键数量">
          {{ cache.total_keys }}
        </el-descriptions-item>
        <!-- 有短期记忆的用户数：来自同一接口的 short_memory_users -->
        <el-descriptions-item label="短期记忆用户数">
          {{ cache.short_memory_users }}
        </el-descriptions-item>
      </el-descriptions>
      <!-- 按用户分组的缓存键数量，数据来自 cacheStats().by_user -->
      <el-table :data="byUserRows" size="small" class="gap" empty-text="暂无缓存">
        <!-- 用户 ID -->
        <el-table-column prop="user_id" label="用户 ID" width="120" />
        <!-- 该用户名下的缓存键数量 -->
        <el-table-column prop="keys" label="缓存键数量" />
      </el-table>
      <el-alert class="tip" type="info" :closable="false" show-icon
                title="缓存键按 cache:rag:answer:{user_id}:{role}:{问题} 命名；清理入口在「知识库」页的“清空我的缓存”。" />
    </el-card>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'           // 导入响应式工具与挂载钩子
import { cacheStats, health, kbStatus, listConversations, listMessages, listRoles } from '../api'   // 只读接口

const loading = ref(false)                                          // 刷新中标记
const svc = reactive({ milvus: false, mysql: false, redis: false })   // 三项服务状态
const kb = reactive({ collection: '', row_count: 0, last_update: '' })   // Milvus 集合信息
const cache = reactive({ total_keys: 0, short_memory_users: 0, by_user: {} })   // Redis 缓存统计

// by_user 是 {"40": 1} 形式的对象，转成数组才能喂给 el-table
const byUserRows = computed(() => Object.entries(cache.by_user || {})   // 取键值对
  .map(([uid, n]) => ({ user_id: uid, keys: n })))                 // 转成 [{user_id, keys}]

// 表清单：前四项来自 db.py 的建表语句；条数只能从现有只读接口推导
const tables = reactive([
  { name: 'users', desc: '用户账号', count: '-', scope: '现有接口不返回用户总数' },
  { name: 'roles', desc: '问答角色', count: '-', scope: 'GET /roles 全量' },
  { name: 'conversations', desc: '会话', count: '-', scope: 'GET /conversation/{user_id}，仅当前用户' },
  { name: 'messages', desc: '会话消息', count: '-', scope: 'GET /message/{conv_id}，仅当前会话' }
])

// 取一张表的行引用，避免重复写 find
const row = (name) => tables.find((t) => t.name === name)

// 汇总加载全部只读数据
async function loadAll() {
  loading.value = true                                             // 置忙
  try {
    // 1) 三项依赖状态
    const h = await health()                                       // 调接口
    Object.assign(svc, h)                                          // 覆盖 milvus / mysql / redis
    // 2) Milvus 集合信息
    Object.assign(kb, await kbStatus())                            // 集合名、条数、更新时间
    // 3) 角色条数
    const roles = await listRoles()                                // 调接口
    row('roles').count = (roles.roles || []).length                // 角色总数
    // 4) 会话条数与当前会话消息条数
    const userId = localStorage.getItem('user_id')                 // 当前用户
    const convs = await listConversations(userId)                  // 会话列表
    row('conversations').count = (convs.conversations || []).length   // 当前用户的会话数
    const convId = localStorage.getItem('conversation_id')         // 本地记录的当前会话
    if (convId) {                                                  // 有当前会话才统计消息
      const msgs = await listMessages(convId)                      // 消息列表
      row('messages').count = (msgs.messages || []).length          // 该会话的消息数
    }
    // 5) Redis 缓存统计
    Object.assign(cache, await cacheStats())                       // 覆盖 total_keys / short_memory_users
  } catch {
    /* 单项失败不影响其余展示，错误已由拦截器统一提示 */
  } finally {
    loading.value = false                                          // 解除置忙
  }
}

onMounted(loadAll)                                                 // 挂载后立即加载
</script>

<style scoped>
/* 页面容器 */
.page {
  max-width: 1000px;        /* 限制宽度 */
  margin: 0 auto;           /* 水平居中 */
}
/* 顶部标题与刷新按钮 */
.head {
  display: flex;            /* 横向排列 */
  justify-content: space-between;   /* 两端对齐 */
  align-items: center;      /* 垂直居中 */
}
/* 卡片间距 */
.gap {
  margin-top: 16px;         /* 上间距 */
}
/* 卡片内说明条间距 */
.tip {
  margin-top: 12px;         /* 上间距 */
}
</style>
