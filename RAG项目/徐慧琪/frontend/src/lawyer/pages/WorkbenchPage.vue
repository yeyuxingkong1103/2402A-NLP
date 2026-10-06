<template>
  <section class="workbench-page">
    <header class="workbench-page__head">
      <!-- 身份取 session（登录响应的 role/team_id + 登录入参的 username）：顶部
           显示「当前是谁、哪个团队」是审计导出的角色显示的前提 -->
      <span class="workbench-page__identity" data-test="identity">
        {{ session.username }} · {{ session.role }} · {{ session.teamId }}
      </span>
      <el-button size="small" data-test="logout" @click="onLogout">退出</el-button>
    </header>
    <div class="workbench-page__modes">
      <!-- 模式切换是本地状态：问答与检索各管各的结果，不互相覆盖 -->
      <el-button data-test="mode-qa" :type="mode === 'qa' ? 'primary' : 'default'" @click="mode = 'qa'">
        问答
      </el-button>
      <el-button data-test="mode-search" :type="mode === 'search' ? 'primary' : 'default'" @click="mode = 'search'">
        检索
      </el-button>
    </div>
    <QuestionBox :loading="loading" :disabled="rateLimited" @submit="submit" @cancel="cancel" />
    <ErrorPanel
      v-if="error"
      data-test="error"
      class="workbench-page__error"
      :error="error"
      @retry="retry"
      @copied="copyText"
      @countdown-end="rateLimited = false"
    />
    <template v-if="mode === 'qa'">
      <!-- 结果卡独立堆叠、最新在上（仅存内存，刷新即清）——与公众侧同一约定 -->
      <div v-for="(r, i) in results" :key="i" class="workbench-page__result" data-test="result">
        <AnswerCard :result="r" copyable @copied="copyText" />
      </div>
    </template>
    <template v-else>
      <div v-if="searchResult" class="workbench-page__search" data-test="search-result">
        <ul class="workbench-page__blocks">
          <li v-for="(b, i) in searchResult.blocks" :key="i" class="workbench-page__block" data-test="block">
            {{ b.article_no_cn ?? `第${b.article_no}条` }} · {{ b.path ?? '—' }} · 相关度 {{ b.rerank_score ?? '—' }}
          </li>
        </ul>
        <!-- exact_nos 是后端的「这次走了精确通路」判据（SearchResponse 注释），
             原样透出条号：律师据此判断检索结果可不可信，前端不加解释性措辞 -->
        <p v-for="no in searchResult.exact_nos" :key="no" class="workbench-page__exact" data-test="exact-no">
          精确置顶：第 {{ no }} 条
        </p>
      </div>
    </template>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ApiError } from '@/core/api/request'
import { qa, search } from '@/core/api/lawyer'
import type { QAAnswer, SearchResponse } from '@/core/api/lawyer'
import { session, logout } from '@/core/useAuth'
import { handleAuthExpired } from '@/lawyer/router'
import AnswerCard from '@/components/AnswerCard.vue'
import ErrorPanel from '@/components/ErrorPanel.vue'
import QuestionBox from '@/components/QuestionBox.vue'

const router = useRouter()
const mode = ref<'qa' | 'search'>('qa')
const results = ref<QAAnswer[]>([])
const searchResult = ref<SearchResponse | null>(null)
const loading = ref(false)
const error = ref<ApiError | null>(null)
// 429 倒计时内禁提交（设计 §五）：与公众 AskPage 同一套状态机——只在 Retry-After
// 存在时置位，ErrorPanel 走到 0 时由 countdown-end 复位
const rateLimited = ref(false)
// 重试重发最后一问（重放语义，不取用户可能改了一半的输入框）
let lastQuestion = ''
// 取消靠把 AbortController 的 signal 透传给当前模式的 API（request.ts 归型成
// aborted）：qa 的 signal 是第二参，search 的在第三参（topK 占着第二参，
// 终审 I4 给 search 增了可选 signal）。两种模式同一套取消口径，检索中按取消
// 不再是无效果的动作
let controller: AbortController | null = null

async function submit(question: string) {
  if (loading.value) return
  // 每次提交都从解锁态重新开始限流状态机：429 倒计时内点重试会先卸载正在倒计时
  // 的 ErrorPanel（error.value = null），countdown-end 随组件消失不再发——不复位
  // 的话这次重试若成功，rateLimited 永不复位，提交按钮永久置灰（只能刷新页面）。
  // 与 AskPage.vue 现实现逐条同构（裁决 A）
  rateLimited.value = false
  lastQuestion = question
  error.value = null
  controller = new AbortController()
  loading.value = true
  try {
    if (mode.value === 'qa') {
      const answer = await qa(question, controller.signal)
      results.value = [answer, ...results.value]
    } else {
      // topK 仍不传（默认值只有后端一处）；只把取消信号透到第三参
      searchResult.value = await search(question, undefined, controller.signal)
    }
  } catch (e) {
    if (e instanceof ApiError && e.kind === 'aborted') {
      // 用户取消：静默，不产生结果也不报错（设计 §五「取消不报错」）
    } else if (e instanceof ApiError) {
      // 专用路由 404 = 会话失效：清 token + 跳登录，不弹普通错误面板。
      // 映射只有 router.ts 一处；返回 false 才落 ErrorPanel
      if (handleAuthExpired(e, router)) return
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

// 复制格式由 AnswerCard 按 FR-7.4 格式化后随事件带出；写剪贴板是页面副作用
// （裁决 2/3）。jsdom 与不安全上下文没有 navigator.clipboard——可选链静默降级
function copyText(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {})
}

function onLogout() {
  logout()
  void router.push('/login')
}
</script>
