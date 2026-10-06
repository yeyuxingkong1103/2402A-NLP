<template>
  <section class="article-page">
    <!-- @copied 接上页面副作用（终审 I2）：不接时复制按钮是静默 no-op，
         request_id 进不了剪贴板（设计 §五「按 id 回查运维日志」） -->
    <ErrorPanel v-if="error" data-test="error" :error="error" @retry="load" @copied="copyText" />
    <!-- 与公众侧同一张卡，多一个复制（FR-7.4：复制规范引用而不是裸正文） -->
    <ArticleText v-else-if="articleData" :article="articleData" copyable @copied="copyText" />
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

// watch 而不是 onMounted：同一实例内换条号要重取，否则第二篇永远显示第一篇
// （静默错页；公众侧同一笔账）
watch(() => [route.params.lawId, route.params.no], load, { immediate: true })

// 复制载荷由 ArticleText 按 FR-7.4 格式化成规范引用后带出；写剪贴板是页面
// 副作用（裁决 2/3）。jsdom 无 clipboard 时可选链静默降级
function copyText(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {})
}
</script>
