<template>
  <section class="login-page">
    <h2 class="login-page__title">律师登录</h2>
    <el-input
      v-model="username"
      class="login-page__input"
      data-test="username"
      placeholder="用户名"
      @keyup.enter="submit"
    />
    <el-input
      v-model="password"
      class="login-page__input"
      data-test="password"
      type="password"
      placeholder="密码"
      show-password
      @keyup.enter="submit"
    />
    <!-- 错误走 ErrorPanel（终审 I3）：后端 message 一律透传（不编造），429 的
         Retry-After 倒计时与 request_id 复制由面板统一承担——登录页此前只有
         一句静态文案，429 后连点仍继续吃 429 -->
    <ErrorPanel
      v-if="error"
      data-test="error"
      class="login-page__error"
      :error="error"
      @retry="submit"
      @copied="copyText"
      @countdown-end="rateLimited = false"
    />
    <el-button
      type="primary"
      data-test="submit"
      :loading="loading"
      :disabled="rateLimited"
      @click="submit"
    >
      登录
    </el-button>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ApiError } from '@/core/api/request'
import { login } from '@/core/useAuth'
import ErrorPanel from '@/components/ErrorPanel.vue'

const route = useRoute()
const router = useRouter()
const username = ref('')
const password = ref('')
const loading = ref(false)
const error = ref<ApiError | null>(null)
// 429 倒计时内禁提交（设计 §五）：与公众 AskPage 同一套状态机——只在
// Retry-After 存在时置位，ErrorPanel 走到 0 时由 countdown-end 复位
const rateLimited = ref(false)

async function submit() {
  if (loading.value) return
  // 每次提交都从解锁态重新开始限流状态机：倒计时内点重试会先卸载正在倒计时的
  // ErrorPanel（error.value = null），countdown-end 随组件消失不再发——不复位
  // 的话这次重试若成功，rateLimited 永不复位，登录按钮永久置灰（只能刷新）。
  // 与 AskPage.vue / WorkbenchPage.vue 现实现逐条同构（终审 I3 的承重面）
  rateLimited.value = false
  error.value = null
  loading.value = true
  try {
    await login(username.value, password.value)
    // 回跳守卫记下的 returnTo（直接访问 /cases 被拦的场景）；没有就落工作台
    const returnTo = typeof route.query.returnTo === 'string' ? route.query.returnTo : '/'
    await router.push(returnTo)
  } catch (e) {
    if (!(e instanceof ApiError)) throw e // 非 ApiError 是编程错误：吞掉会停在「点了没反应」
    error.value = e
    if (e.retryAfter !== null) rateLimited.value = true
  } finally {
    loading.value = false
  }
}

// request_id 的复制副作用（设计 §五，终审 I2 同批）：写剪贴板是页面的事；
// jsdom 与非安全上下文没有 navigator.clipboard——可选链静默降级
function copyText(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {})
}
</script>
