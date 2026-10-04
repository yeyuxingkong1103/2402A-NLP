<template>
  <section class="error-panel">
    <!-- message 一律透传后端（不编造）：后端文案就是用户该看到的那句 -->
    <p class="error-panel__message">{{ error.message }}</p>
    <p v-if="error.code" class="error-panel__code">错误代码：{{ error.code }}</p>
    <!-- 倒计时起点是 Retry-After 的秒数（request.ts 解析），不是前端自估 -->
    <p v-if="countdown !== null" class="error-panel__countdown" data-test="countdown">
      {{ countdown }} 秒后可重试
    </p>
    <!-- request_id 是「按 id 回查运维日志」的唯一把手（交付说明 §四），必须显示；
         复制按钮只把 id 随事件带出，写剪贴板是页面的副作用（裁决 1） -->
    <p v-if="error.requestId" class="error-panel__request-id" data-test="request-id">
      请求编号：{{ error.requestId }}
      <el-button size="small" data-test="copy-id" @click="copyRequestId">复制</el-button>
    </p>
    <el-button type="primary" data-test="retry" @click="emit('retry')">重试</el-button>
  </section>
</template>

<script setup lang="ts">
import { onUnmounted, ref, watch } from 'vue'
// import type：ApiError 是构造器类型，但组件只把它当结构用（message/code/...），
// 不 new 不 instanceof——编译期擦除即可，不制造运行时模块边（裁决 1）
import type { ApiError } from '@/core/api/request'

const props = defineProps<{ error: ApiError }>()
// countdown-end（裁决 A）：倒计时走到 0 的通知，页面据它解除「倒计时内禁提交」。
// 事件在每个倒计时周期恰好发一次——重复发会让页面反复复位，漏发会让提交
// 按钮永远禁用；两者都没有别的红灯
const emit = defineEmits<{ retry: []; copied: [text: string]; 'countdown-end': [] }>()

const countdown = ref<number | null>(null)
// 倒计时是组件里唯一会在卸载后继续跑的东西；error 是可变的 prop（重试后换成
// 新错误），所以用 watch immediate 起步：换错误 = 重取 Retry-After 重新计时。
let timer: ReturnType<typeof setInterval> | null = null
function stopTimer() {
  if (timer !== null) {
    clearInterval(timer)
    timer = null
  }
}
function startCountdown() {
  stopTimer()
  countdown.value = props.error.retryAfter
  if (countdown.value === null) return
  // Retry-After: 0 = 后端说「现在就可以重试」：没有一秒要等，立即通知一次
  if (countdown.value <= 0) {
    countdown.value = 0
    emit('countdown-end')
    return
  }
  timer = setInterval(() => {
    const left = (countdown.value ?? 1) - 1
    countdown.value = left > 0 ? left : 0
    if (left <= 0) {
      stopTimer() // 到 0 即停：不为一个不再变化的数字空转
      // 恰好一次：停表后不再有 tick；重复换 error 重启时 stopTimer 先清旧表
      emit('countdown-end')
    }
  }, 1000)
}
watch(() => props.error, startCountdown, { immediate: true })
onUnmounted(stopTimer)

function copyRequestId() {
  if (props.error.requestId != null) emit('copied', props.error.requestId)
}
</script>
