<template>
  <section class="nav-page">
    <h2 class="nav-page__title">法条浏览</h2>
    <!-- @copied 必须接上（终审 I2）：不接监听时复制按钮点击后毫无动静，
         request_id 无从进入剪贴板（设计 §五：按 id 回查运维日志） -->
    <ErrorPanel v-if="error" :error="error" @retry="load" @copied="copyText" />
    <!-- 树只在拿到响应后渲染：NavTree 的 nodes 是必需 prop，未加载完就挂一棵
         空树，用户看到的是「民法典一条都没有」，与请求失败不可区分 -->
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
// 跳文章详情要 law_id。取第一棵树的 law_id 而不是硬编码 'minfadian'：
// 多法时路径参数必须来自数据，来自常量会把别家法的条号挂到民法典路径下
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
  // 具名路由 + params：路径模式只在 router.ts 写一次，页面不复制路径字面量
  void router.push({ name: 'article', params: { lawId, no: String(articleNo) } })
}

// request_id 的复制副作用（设计 §五）：组件只 emit，写剪贴板是页面的事；
// jsdom 与非安全上下文没有 navigator.clipboard——可选链静默降级
function copyText(text: string) {
  navigator.clipboard?.writeText(text).catch(() => {})
}

onMounted(load)
</script>
