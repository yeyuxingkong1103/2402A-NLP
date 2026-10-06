<template>
  <section class="article-page">
    <!-- @copied 必须接上（终审 I2）：ErrorPanel 的按钮只把 request_id 随事件带出，
         不接监听时点击是静默 no-op——设计 §五「按 id 回查运维日志」在这页落空 -->
    <ErrorPanel v-if="error" :error="error" @retry="load" @copied="copyText" />
    <ArticleText v-else-if="articleData" :article="articleData" />
  </section>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ApiError } from '@/core/api/request'
import { article } from '@/core/api/public'
import type { ArticleResponse } from '@/core/api/public'
import ArticleText from '@/components/ArticleText.vue'
import ErrorPanel from '@/components/ErrorPanel.vue'

const route = useRoute()
const articleData = ref<ArticleResponse | null>(null)
const error = ref<ApiError | null>(null)

async function load() {
  const lawId = String(route.params.lawId ?? '')
  const no = String(route.params.no ?? '')
  if (!lawId || !no) return
  error.value = null
  articleData.value = null
  try {
    articleData.value = await article(lawId, no)
  } catch (e) {
    if (e instanceof ApiError) error.value = e
    else throw e
  }
}

// watch 而不是 onMounted：/law/x/articles/1 → /articles/2 会复用同一个组件
// 实例，只在挂载时取一次的话第二篇永远显示第一篇的正文（静默错页）
watch(() => [route.params.lawId, route.params.no], load, { immediate: true })

// request_id 的复制副作用（设计 §五，与 AskPage 同形）：写剪贴板是页面的事，
// 组件只 emit；jsdom 与非安全上下文没有 navigator.clipboard——可选链静默降级
function copyText(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {})
}
</script>
