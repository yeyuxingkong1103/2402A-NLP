<template>
  <section class="question-box">
    <textarea
      class="question-box__input"
      data-test="input"
      :value="question"
      :maxlength="MAX_QUESTION_CHARS"
      :disabled="loading"
      placeholder="请用日常语言描述你的问题…"
      @input="onInput"
    />
    <!-- 计数与上限同屏：用户不必等到提交被拒才知道还有多少富余 -->
    <p class="question-box__count">{{ question.length }} / {{ MAX_QUESTION_CHARS }}</p>
    <!-- 超限提示只在真的超限时出现：maxlength 在真实浏览器里挡键盘输入，但
         粘贴/输入法组合/程序化赋值仍可能越界——在后端 400 之前先本地拦住 -->
    <p v-if="overLimit" class="question-box__over" data-test="over-limit">
      已超出 {{ MAX_QUESTION_CHARS }} 字上限，请删减后再提交
    </p>
    <p v-if="loading" class="question-box__elapsed" data-test="elapsed">
      已等待 {{ elapsed }} 秒
    </p>
    <div class="question-box__actions">
      <!-- loading 时只剩取消：正在跑的请求不能被第二次提交覆盖 -->
      <el-button v-if="loading" data-test="cancel" @click="emit('cancel')">取消</el-button>
      <el-button
        v-else
        class="question-box__submit"
        data-test="submit"
        type="primary"
        :disabled="!canSubmit"
        @click="onSubmit"
      >提问</el-button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { MAX_QUESTION_CHARS } from '@/core/config'

// disabled（裁决 A：429 倒计时内禁提交）：只压提交按钮，输入框保持可编辑——
// 用户可以先改问题，等倒计时结束直接提交，而不是干等或丢失已输入的内容
const props = defineProps<{ loading?: boolean; disabled?: boolean }>()
// submit 的载荷是 trim 后的问句（页面直接拿去调 API，不再加工）
const emit = defineEmits<{ submit: [question: string]; cancel: [] }>()

const question = ref('')
const elapsed = ref(0)

function onInput(event: Event) {
  question.value = (event.target as HTMLTextAreaElement).value
}

// 超限按原始长度判：trim 解决不了这个——500 个内部空格也是 500 字
const overLimit = computed(() => question.value.length > MAX_QUESTION_CHARS)
// 全空白 = 没有内容可问（后端实测会把空问句判成无效输入）；禁用而不是发了等 400
const canSubmit = computed(() =>
  !props.loading && !props.disabled && !overLimit.value && question.value.trim().length > 0)

function onSubmit() {
  if (!canSubmit.value) return
  // 发出去的是用户看到的内容去掉首尾排版空白：首尾空格会改变 embedding 的输入，
  // 而这部分空白对用户没有语义
  emit('submit', question.value.trim())
}

// 「已等待 N 秒」的 interval：QA 实测一发 13~24s，没有计时用户会以为卡死。
// 这是组件里唯一会在卸载后继续跑的东西，loading 结束与卸载都必须清掉。
let timer: ReturnType<typeof setInterval> | null = null
function stopTimer() {
  if (timer !== null) {
    clearInterval(timer)
    timer = null
  }
}
watch(() => props.loading, (on) => {
  stopTimer()
  elapsed.value = 0
  if (on) timer = setInterval(() => { elapsed.value += 1 }, 1000)
}, { immediate: true })
onUnmounted(stopTimer)
</script>
