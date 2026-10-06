<template>
  <ul v-if="citations.length" class="citation-list">
    <li v-for="(c, i) in citations" :key="i" class="citation-list__item">
      <span class="citation-list__no">{{ articleLabel(c.article) }}</span>
      <span class="citation-list__quote">“{{ c.quote }}”</span>
      <!-- 必须走 RouterLink，不能退回普通 a（终审 I1）：普通锚点是整页导航，
           根路径 '/law/…' 在同源下归属**公众入口**——律师点「查看原文」会被送出
           律师壳（公众条文页没有复制按钮，FR-7.4 在浏览器里不可用）。RouterLink
           在当前入口自己的路由表内解析（两入口都有 article 路由），渲染出的 href
           仍是 citationUrl(c) 的路径字面量，点击不产生整页跳转。 -->
      <RouterLink class="citation-list__link" :to="citationUrl(c)">查看原文</RouterLink>
    </li>
  </ul>
</template>

<script setup lang="ts">
// 引用列表：条号 + 引文 + 跳转。类型从 public.ts 引（import type 擦除），
// 组件可被两入口共享而不产生运行时模块边（裁决 1）。
import type { Citation } from '@/core/api/public'
import { articleLabel, citationUrl } from '@/core/format'

defineProps<{ citations: Citation[] }>()
</script>
