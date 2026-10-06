<template>
  <section class="cases-page">
    <h2 class="cases-page__title">历史案件</h2>
    <!-- 这一页是接口位联调证明：后端 501（未实现）的 reason 原样展示，
         不编造「暂无案件」之类的壳文案——真正的判据是后端说了什么 -->
    <!-- @copied 必须接上（终审 I2）：不接监听时复制按钮是静默 no-op，
         request_id 进不了剪贴板（设计 §五「按 id 回查运维日志」） -->
    <ErrorPanel v-if="error" data-test="error" :error="error" @retry="load" @copied="copyText" />
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ApiError } from '@/core/api/request'
import { casesSearch } from '@/core/api/lawyer'
import { handleAuthExpired } from '@/lawyer/router'
import ErrorPanel from '@/components/ErrorPanel.vue'

const router = useRouter()
const error = ref<ApiError | null>(null)

async function load() {
  error.value = null
  try {
    await casesSearch()
  } catch (e) {
    if (!(e instanceof ApiError)) throw e
    // 专用路由 404 = 会话失效 → 登录页；501 等其余错误落面板显示后端 message
    if (handleAuthExpired(e, router)) return
    error.value = e
  }
}

onMounted(load)

// request_id 的复制副作用（设计 §五）：组件只 emit，写剪贴板是页面的事；
// jsdom 与非安全上下文没有 navigator.clipboard——可选链静默降级
function copyText(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {})
}
</script>
