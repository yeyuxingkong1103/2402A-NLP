<!-- 公众入口的壳：页头标题 + 路由内容 + 常驻页脚（设计 §四）。页脚放在壳里
     而不是每个页面自己挂：FR-8.4 的免责句要求「每页常驻」，靠页面自觉就会
     漏一个页面——漏掉的表现是那一页没有任何免责，没有红灯只有合规缺口 -->
<template>
  <header class="app-header">
    <h1 class="app-header__title">法律咨询 · 民法典问答</h1>
  </header>
  <main class="app-main">
    <router-view />
  </main>
  <SiteFooter />
</template>

<script setup lang="ts">
import SiteFooter from '@/components/SiteFooter.vue'
</script>
