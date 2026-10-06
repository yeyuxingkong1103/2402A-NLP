<template>
  <section class="ask-page">
    <h2 class="ask-page__title">民法典问答</h2>
    <QuestionBox
      :loading="loading"
      :disabled="rateLimited"
      @submit="submit"
      @cancel="cancel"
    />
    <ErrorPanel
      v-if="error"
      class="ask-page__error"
      :error="error"
      @retry="retry"
      @copied="copyText"
      @countdown-end="rateLimited = false"
    />
    <!-- 结果卡独立堆叠、最新在上（仅存内存，刷新即清）：每次提问是一个独立
         结果对象而不是替换同一张卡——上一问的解读在下一问之后仍可回看 -->
    <div v-for="(r, i) in results" :key="i" class="ask-page__result" data-test="result">
      <AnswerCard :result="r" />
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ApiError } from '@/core/api/request'
import { publicQa } from '@/core/api/public'
import type { PublicQAAnswer } from '@/core/api/public'
import AnswerCard from '@/components/AnswerCard.vue'
import ErrorPanel from '@/components/ErrorPanel.vue'
import QuestionBox from '@/components/QuestionBox.vue'

const results = ref<PublicQAAnswer[]>([])
const loading = ref(false)
const error = ref<ApiError | null>(null)
// 429 倒计时内禁提交（设计 §五）：只在 Retry-After 存在时置位，由 ErrorPanel
// 走到 0 时发来的 countdown-end 复位（裁决 A 的落法，Task 6 复用同一套）
const rateLimited = ref(false)
// 重试重发的是「最后一问」，不是用户此刻可能改了一半的输入框内容——
// 错误面板的重试按钮语义是重放，不是重新编辑
let lastQuestion = ''
// 取消靠把 AbortController 的 signal 透传给 publicQa（request.ts 会归型成
// aborted），页面只负责按下去；没有在跑的请求时按不到取消按钮
let controller: AbortController | null = null

async function submit(question: string) {
  if (loading.value) return
  // 每次提交都从解锁态重新开始限流状态机：429 倒计时内点重试会先把正在倒计时的
  // ErrorPanel 卸载（error.value = null），countdown-end 随组件消失不再发——不复位
  // 的话这次重试若成功，rateLimited 永不复位，提交按钮永久置灰（只能刷新页面）。
  // 复位后若再吃 429，catch 会重新置位，新 ErrorPanel 的倒计时照常兜底
  rateLimited.value = false
  lastQuestion = question
  error.value = null
  controller = new AbortController()
  loading.value = true
  try {
    const answer = await publicQa(question, controller.signal)
    results.value = [answer, ...results.value]
  } catch (e) {
    if (e instanceof ApiError && e.kind === 'aborted') {
      // 用户取消：静默，不产生结果卡也不报错（设计 §五「取消不报错」）——
      // 输入框里的问句本来就没被清掉，用户可以直接改了再问
    } else if (e instanceof ApiError) {
      error.value = e
      if (e.retryAfter !== null) rateLimited.value = true
    } else {
      throw e // 非 ApiError 是编程错误：吞掉会让页面停在「什么都没发生」
    }
  } finally {
    loading.value = false
    controller = null
  }
}

function cancel() {
  controller?.abort()
}

function retry() {
  if (lastQuestion) void submit(lastQuestion)
}

// request_id 的复制副作用（设计 §五：按 id 回查运维日志）。jsdom 与
// 非安全上下文没有 navigator.clipboard——可选链短路，静默降级；
// 错误面板的可见文本不依赖这一步
function copyText(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {})
}
</script>
