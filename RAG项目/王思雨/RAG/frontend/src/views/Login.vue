<!-- 第 9 步：登录 / 注册页面 -->
<template>
  <!-- 页面居中容器 -->
  <div class="login-wrap">
    <!-- 登录注册卡片 -->
    <el-card class="login-card">
      <!-- 卡片标题 -->
      <template #header>RAG 电力维修问答助手</template>
      <!-- 两个页签：登录与注册 -->
      <el-tabs v-model="tab">
        <!-- 登录页签 -->
        <el-tab-pane label="登录" name="login">
          <!-- 登录表单 -->
          <el-form :model="loginForm" label-width="70px">
            <!-- 用户名输入 -->
            <el-form-item label="用户名">
              <el-input v-model="loginForm.username" placeholder="请输入用户名" />
            </el-form-item>
            <!-- 密码输入，show-password 可切换明文 -->
            <el-form-item label="密码">
              <el-input v-model="loginForm.password" type="password" show-password
                        placeholder="请输入密码" @keyup.enter="doLogin" />
            </el-form-item>
            <!-- 登录按钮 -->
            <el-form-item>
              <el-button type="primary" :loading="busy" @click="doLogin">登录</el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
        <!-- 注册页签 -->
        <el-tab-pane label="注册" name="register">
          <!-- 注册表单 -->
          <el-form :model="regForm" label-width="70px">
            <!-- 用户名输入 -->
            <el-form-item label="用户名">
              <el-input v-model="regForm.username" placeholder="1-64 个字符" />
            </el-form-item>
            <!-- 密码输入，后端要求至少 6 位 -->
            <el-form-item label="密码">
              <el-input v-model="regForm.password" type="password" show-password
                        placeholder="至少 6 位" />
            </el-form-item>
            <!-- 确认密码：仅前端比对，防止输错 -->
            <el-form-item label="确认密码">
              <el-input v-model="regForm.confirm" type="password" show-password
                        placeholder="再输一次密码" @keyup.enter="doRegister" />
            </el-form-item>
            <!-- 注册按钮 -->
            <el-form-item>
              <el-button type="success" :loading="busy" @click="doRegister">注册</el-button>
            </el-form-item>
          </el-form>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'          // 导入响应式工具
import { useRouter } from 'vue-router'       // 导入路由，用于登录后跳转
import { ElMessage } from 'element-plus'     // 导入消息提示
import { login, register } from '../api'     // 导入登录与注册接口

const router = useRouter()                                          // 路由实例
const tab = ref('login')                                            // 当前页签，默认登录
const busy = ref(false)                                             // 请求中标记，防止重复点击
const loginForm = reactive({ username: '', password: '' })          // 登录表单数据
const regForm = reactive({ username: '', password: '', confirm: '' })   // 注册表单数据

// 登录成功后统一写入本地存储并跳转到角色选择页
function saveAndGo(data) {                                          // data 为后端返回体
  localStorage.setItem('user_id', String(data.user_id))             // 用户编号，后面所有请求都要带
  localStorage.setItem('username', data.username)                   // 用户名，导航栏展示用
  localStorage.removeItem('role_id')                                // 清掉旧角色，避免跨账号串用
  localStorage.removeItem('role_name')                              // 同上
  localStorage.removeItem('conversation_id')                        // 清掉旧会话，避免跨账号串用
  ElMessage.success(`欢迎，${data.username}`)                        // 提示成功
  router.push('/role')                                              // 跳角色选择页
}

// 登录处理
async function doLogin() {
  if (!loginForm.username || !loginForm.password) {                 // 必填校验
    ElMessage.warning('请输入用户名和密码')                          // 提示
    return                                                          // 中止
  }
  busy.value = true                                                 // 置忙
  try {
    saveAndGo(await login(loginForm.username, loginForm.password))   // 调接口并处理结果
  } finally {
    busy.value = false                                              // 无论成败都解除置忙
  }
}

// 注册处理
async function doRegister() {
  if (!regForm.username || !regForm.password) {                     // 必填校验
    ElMessage.warning('请输入用户名和密码')                          // 提示
    return                                                          // 中止
  }
  if (regForm.password !== regForm.confirm) {                       // 两次密码一致校验
    ElMessage.warning('两次输入的密码不一致')                        // 提示
    return                                                          // 中止
  }
  busy.value = true                                                 // 置忙
  try {
    const data = await register(regForm.username, regForm.password)   // 调注册接口
    ElMessage.success('注册成功，请登录')                            // 提示
    loginForm.username = data.username                              // 把用户名带到登录页签
    loginForm.password = ''                                         // 密码留空让用户重输
    tab.value = 'login'                                             // 切回登录页签
  } finally {
    busy.value = false                                              // 解除置忙
  }
}
</script>

<style scoped>
/* 居中容器：卡片水平垂直居中 */
.login-wrap {
  display: flex;                 /* 弹性布局 */
  justify-content: center;       /* 水平居中 */
  padding-top: 80px;             /* 距顶部留白 */
}
/* 卡片宽度 */
.login-card {
  width: 420px;                  /* 固定宽度，避免过宽 */
}
</style>
