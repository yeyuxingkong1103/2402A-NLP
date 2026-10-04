<template>
  <section class="nav-page">
    <h2 class="nav-page__title">法条浏览</h2>
    <!-- 错误分支有判据（Task 5 审查回灌）：错误 → ErrorPanel 渲染且 retry 重发，
         不是静默留一棵空树 -->
    <ErrorPanel v-if="error" data-test="error" :error="error" @retry="load" @copied="copyText" />
    <NavTree v-else-if="nodes" :nodes="nodes" @select="openArticle" />
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ApiError } from '@/core/api/request'
import { nav } from '@/core/api/public'
import type { NavNode } from '@/core/api/public'
import ErrorPanel from '@/components/ErrorPanel.vue'
import NavTree from '@/components/NavTree.vue'

const router = useRouter()
const nodes = ref<NavNode[] | null>(null)
const error = ref<ApiError | null>(null)
// law_id 取数据不写死（Task 5 审查回灌）：常量作路径参数会把别家法的条号
// 挂到民法典路径下；具名路由 push，路径模式只在 router.ts 写一次
let lawId = ''

async function load() {
  error.value = null
  nodes.value = null
  try {
    const res = await nav()
    const first = res.laws[0]
    nodes.value = first?.nodes ?? []
    lawId = first?.law_id ?? ''
  } catch (e) {
    if (e instanceof ApiError) error.value = e
    else throw e
  }
}

function openArticle(articleNo: number) {
  if (!lawId) return
  void router.push({ name: 'article', params: { lawId, no: String(articleNo) } })
}

// request_id 的复制副作用（设计 §五，终审 I2 补齐）：组件只 emit，写剪贴板
// 是页面的事；jsdom 与非安全上下文没有 navigator.clipboard——可选链静默降级
function copyText(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {})
}

onMounted(load)
</script>
