<!-- 第 9 步：角色选择页面 -->
<template>
  <!-- 页面容器 -->
  <div class="page">
    <!-- 页面标题 -->
    <h2>请选择角色</h2>
    <!-- 角色卡片行 -->
    <div class="cards">
      <!-- 逐个角色渲染一张卡片 -->
      <el-card v-for="role in roles" :key="role.role_id" class="card"
               :class="{ active: role.role_id === activeId }" shadow="hover">
        <!-- 角色名 -->
        <div class="role-name">
          {{ role.role_name }}
          <!-- 默认推荐角色角标 -->
          <el-tag v-if="role.role_name === 'power_repair_expert'" type="success" size="small">
            推荐
          </el-tag>
        </div>
        <!-- 角色说明 -->
        <p class="desc">{{ role.description || '暂无说明' }}</p>
        <!-- 选择按钮 -->
        <el-button type="primary" @click="choose(role)">选择该角色</el-button>
      </el-card>
    </div>
    <!-- 加载中提示 -->
    <el-empty v-if="!loading && roles.length === 0" description="暂无可用角色，请确认后端已初始化角色" />
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'          // 导入响应式工具与挂载钩子
import { useRouter } from 'vue-router'        // 导入路由
import { ElMessage } from 'element-plus'      // 导入消息提示
import { listRoles } from '../api'            // 导入角色列表接口

const router = useRouter()                                        // 路由实例
const roles = ref([])                                             // 角色列表
const activeId = ref(null)                                        // 当前高亮的角色编号
const loading = ref(true)                                         // 加载标记

// 加载角色列表，并默认高亮电力维修专家
async function loadRoles() {
  loading.value = true                                            // 置为加载中
  try {
    const data = await listRoles()                                // 调接口
    roles.value = data.roles || []                                // 存列表
    const saved = Number(localStorage.getItem('role_id'))         // 上次选过的角色
    const preferred = roles.value.find((r) => r.role_name === 'power_repair_expert')   // 默认推荐角色
    // 高亮顺序：上次选的 > 电力维修专家 > 第一个
    activeId.value = saved || preferred?.role_id || roles.value[0]?.role_id || null
  } finally {
    loading.value = false                                         // 解除加载中
  }
}

// 选择角色：写入本地存储后进入对话页
function choose(role) {
  activeId.value = role.role_id                                   // 高亮当前选择
  localStorage.setItem('role_id', String(role.role_id))           // 角色编号，问答请求要带
  localStorage.setItem('role_name', role.role_name)               // 角色名，导航栏展示用
  localStorage.removeItem('conversation_id')                      // 换角色后另开会话，避免串上下文
  ElMessage.success(`已选择：${role.role_name}`)                   // 提示
  router.push('/chat')                                            // 跳转对话页
}

onMounted(loadRoles)                                              // 挂载后立即加载
</script>

<style scoped>
/* 页面容器 */
.page {
  max-width: 1000px;             /* 限制宽度 */
  margin: 0 auto;                /* 水平居中 */
}
/* 卡片横向排列并自动换行 */
.cards {
  display: flex;                 /* 弹性布局 */
  flex-wrap: wrap;               /* 放不下就换行 */
  gap: 16px;                     /* 卡片间距 */
}
/* 单张角色卡片 */
.card {
  width: 300px;                  /* 固定宽度，多张自动排开 */
  border: 2px solid transparent; /* 预留边框，选中时变色不跳动 */
}
/* 选中态：加绿色边框突出 */
.card.active {
  border-color: #67c23a;         /* 选中边框色 */
}
/* 角色名一行 */
.role-name {
  font-size: 16px;               /* 字号 */
  font-weight: bold;             /* 加粗 */
  margin-bottom: 8px;            /* 与说明留白 */
}
/* 角色说明文字 */
.desc {
  color: #606266;                /* 次要文字颜色 */
  min-height: 40px;              /* 占位高度，卡片高度一致 */
}
</style>
