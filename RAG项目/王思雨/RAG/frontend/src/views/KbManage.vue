<!-- 第 9 步：知识库管理页面 -->
<template>
  <!-- 页面容器 -->
  <div class="page">
    <!-- 知识库状态卡片 -->
    <el-card>
      <!-- 标题带刷新按钮 -->
      <template #header>
        <span>知识库状态</span>
        <el-button link type="primary" class="right" @click="loadStatus">刷新</el-button>
      </template>
      <!-- 状态描述列表 -->
      <el-descriptions :column="1" border>
        <!-- Milvus 集合名 -->
        <el-descriptions-item label="集合名">{{ status.collection || '-' }}</el-descriptions-item>
        <!-- 向量条数，正常应为 724 -->
        <el-descriptions-item label="向量条数">
          <b class="count">{{ status.row_count }}</b>
        </el-descriptions-item>
        <!-- 增强结果文件更新时间 -->
        <el-descriptions-item label="最后更新">{{ status.last_update || '-' }}</el-descriptions-item>
      </el-descriptions>

      <!-- 操作按钮区 -->
      <div class="actions">
        <!-- 重建知识库：后端后台执行，不会立刻完成 -->
        <el-button type="primary" :loading="rebuilding" @click="doRebuild">
          重建知识库
        </el-button>
        <!-- 清空当前用户的缓存 -->
        <el-button type="warning" :loading="clearing" @click="doClear">
          清空我的缓存
        </el-button>
      </div>

      <!-- 重建提示 -->
      <el-alert v-if="rebuildMsg" class="tip" type="warning" :closable="false" show-icon
                :title="rebuildMsg" />
      <!-- 清理结果明细 -->
      <el-alert v-if="cacheMsg" class="tip" type="success" :closable="false" show-icon
                :title="cacheMsg" />
    </el-card>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'      // 导入响应式工具与挂载钩子
import { ElMessage } from 'element-plus'            // 导入消息提示
import { clearCache, kbStatus, rebuildKb } from '../api'   // 导入知识库与缓存接口

// 知识库状态：集合名、条数、更新时间
const status = reactive({ collection: '', row_count: 0, last_update: '' })
const rebuilding = ref(false)                       // 重建请求中
const clearing = ref(false)                         // 清理请求中
const rebuildMsg = ref('')                          // 重建提示文字
const cacheMsg = ref('')                            // 清理结果文字

// 拉取知识库状态
async function loadStatus() {
  const data = await kbStatus()                     // 调接口
  status.collection = data.collection               // 集合名
  status.row_count = data.row_count                 // 向量条数
  status.last_update = data.last_update             // 更新时间
}

// 触发知识库重建：后端后台跑，本接口立即返回
async function doRebuild() {
  rebuilding.value = true                           // 置忙
  try {
    const data = await rebuildKb()                  // 调接口
    rebuildMsg.value = data.msg || '重建任务已提交'    // 展示后端提示
    ElMessage.success('重建任务已提交')               // 额外提示
  } finally {
    rebuilding.value = false                        // 解除置忙
  }
}

// 清空当前用户的短期记忆与答案缓存
async function doClear() {
  clearing.value = true                             // 置忙
  try {
    const userId = localStorage.getItem('user_id')  // 当前用户编号
    const data = await clearCache(userId)           // 调接口
    // 拼出清理明细：缓存键个数 + 短期记忆是否清除
    cacheMsg.value = `已清理缓存键 ${data.cache_keys} 个，短期记忆${data.short_memory ? '已' : '未'}清除`
    ElMessage.success('缓存已清理')                  // 提示
  } finally {
    clearing.value = false                          // 解除置忙
  }
}

onMounted(() => { loadStatus().catch(() => {}) })   // 挂载后加载状态，失败由拦截器提示
</script>

<style scoped>
/* 页面容器 */
.page {
  max-width: 800px;      /* 限制宽度 */
  margin: 0 auto;        /* 水平居中 */
}
/* 标题栏右侧按钮 */
.right {
  float: right;          /* 浮动到右侧 */
}
/* 条数突出显示 */
.count {
  color: #409eff;        /* 主题蓝 */
  font-size: 18px;       /* 放大字号 */
}
/* 按钮区 */
.actions {
  margin-top: 16px;      /* 与描述列表留白 */
}
/* 提示条间距 */
.tip {
  margin-top: 12px;      /* 上间距 */
}
</style>
