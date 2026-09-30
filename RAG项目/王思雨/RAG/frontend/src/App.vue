<!-- 第 9 步：根组件，顶部导航 + 路由出口 -->
<template>
  <!-- 外层容器：占满整屏 -->
  <div class="app-root">
    <!-- 顶部导航栏：登录页不显示 -->
    <header v-if="!isLoginPage" class="app-header">
      <!-- 左侧：站点名 -->
      <span class="brand">RAG 电力维修问答助手</span>
      <!-- 中间：导航菜单，开启 router 模式后点击即跳转路由 -->
      <el-menu :default-active="route.path" mode="horizontal" router class="nav-menu">
        <!-- 逐项渲染菜单，key 用路由路径保证唯一 -->
        <el-menu-item v-for="item in menus" :key="item.path" :index="item.path">
          <!-- 菜单图标 -->
          <el-icon><component :is="item.icon" /></el-icon>
          <!-- 菜单文字 -->
          <span>{{ item.label }}</span>
        </el-menu-item>
      </el-menu>
      <!-- 右侧：登录后显示角色名与用户名，未登录只显示“未登录” -->
      <span class="user-box">
        <template v-if="state.logged">
          <!-- 角色标签：当前使用的角色 -->
          <el-tag type="success" size="small">{{ state.roleName }}</el-tag>
          <!-- 用户名 -->
          <span class="username">{{ state.username }}</span>
        </template>
        <template v-else>
          <!-- 未登录提示 -->
          <el-tag type="info" size="small">未登录</el-tag>
        </template>
      </span>
    </header>
    <!-- 路由出口：各页面组件渲染到这里 -->
    <main class="app-main">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { computed } from 'vue'                    // 导入计算属性
import { useRoute } from 'vue-router'             // 导入当前路由对象

const route = useRoute()                          // 取当前路由，用于高亮菜单与判断是否为登录页

// 登录页不显示导航栏
const isLoginPage = computed(() => route.path === '/login')   // 路径为 /login 时隐藏导航

// 导航菜单项：文字、路由路径与图标名一一对应
const menus = [
  { label: '对话', path: '/chat', icon: 'ChatDotRound' },      // 智能问答
  { label: '历史', path: '/history', icon: 'Clock' },          // 历史记录
  { label: '知识库', path: '/kb', icon: 'Collection' },        // 知识库管理
  { label: '上传', path: '/upload', icon: 'UploadFilled' },    // 资料上传
  { label: '评测', path: '/eval', icon: 'DataAnalysis' },      // 评测结果
  { label: '后台', path: '/admin', icon: 'Monitor' },          // 后台状态
  { label: '用户', path: '/user', icon: 'User' }               // 用户中心
]

// 登录态：教学版直接读 localStorage；依赖 route.fullPath 以便登录跳转后自动刷新
const state = computed(() => {
  void route.fullPath                                          // 建立响应依赖：路由变化即重算
  return {
    logged: !!localStorage.getItem('user_id'),                 // 有用户编号即视为已登录
    username: localStorage.getItem('username') || '',          // 用户名
    roleName: localStorage.getItem('role_name') || '未选角色'   // 角色名，未选时给默认文字
  }
})
</script>

<style scoped>
/* 外层容器：使用常见中文字体 */
.app-root {
  font-family: "Microsoft YaHei", Arial, sans-serif;   /* 中文字体 */
  min-height: 100vh;                                   /* 至少占满一屏 */
  background: #f5f7fa;                                 /* 浅灰背景，突出卡片 */
}
/* 顶部导航栏：横向排列并对齐 */
.app-header {
  display: flex;                                       /* 弹性布局 */
  align-items: center;                                 /* 垂直居中 */
  gap: 16px;                                           /* 元素间距 */
  padding: 0 16px;                                     /* 左右留白 */
  background: #fff;                                    /* 白色背景 */
  border-bottom: 1px solid #e4e7ed;                    /* 底部分割线 */
}
/* 站点名样式 */
.brand {
  font-size: 18px;                                     /* 字号 */
  font-weight: bold;                                   /* 加粗 */
  white-space: nowrap;                                 /* 不换行 */
}
/* 菜单占满中间剩余空间 */
.nav-menu {
  flex: 1;                                             /* 撑开中间区域 */
  border-bottom: none;                                 /* 去掉自带下划线 */
}
/* 右侧用户区 */
.user-box {
  display: flex;                                       /* 横向排列 */
  align-items: center;                                 /* 垂直居中 */
  gap: 8px;                                            /* 标签与用户名间距 */
  white-space: nowrap;                                 /* 不换行 */
}
/* 用户名文字 */
.username {
  font-weight: bold;                                   /* 加粗突出 */
}
/* 页面主体区域 */
.app-main {
  padding: 16px;                                       /* 四周留白 */
}
</style>
