<!-- 律师入口的壳：页头 + 路由内容 + 页脚。身份（角色/团队）与退出在工作台页顶
     （设计 §四）而不在壳里——登录页也共用这个壳，壳不认识 session 更简单，
     也避免登录页出现一条「退出」 -->
<template>
  <header class="app-header">
    <h1 class="app-header__title">律师工作台</h1>
  </header>
  <main class="app-main">
    <router-view />
  </main>
  <SiteFooter />
</template>

<script setup lang="ts">
import SiteFooter from '@/components/SiteFooter.vue'
</script>
