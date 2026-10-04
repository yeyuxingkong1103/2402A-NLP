<template>
  <section v-if="lawyers.length" class="lawyer-cards">
    <h4 class="lawyer-cards__title">推荐律师</h4>
    <ul class="lawyer-cards__list">
      <li v-for="(l, i) in lawyers" :key="i" class="lawyer-cards__item">
        <div class="lawyer-cards__row">
          <span class="lawyer-cards__name">{{ l.name }}</span>
          <!-- 演示数据必须显著标注（风险 11：不可构成虚假宣传）——角标是否出现
               只认 demo 字段本身，不认「名字看起来像真的」这类猜测 -->
          <el-tag v-if="l.demo === true" class="lawyer-cards__demo" size="small" type="warning">
            示例数据
          </el-tag>
        </div>
        <p class="lawyer-cards__org">{{ l.org }}</p>
        <p class="lawyer-cards__field">领域：{{ l.field }}</p>
        <p class="lawyer-cards__contact">{{ l.contact_hint }}</p>
      </li>
    </ul>
  </section>
</template>

<script setup lang="ts">
import type { LawyerCard } from '@/core/api/public'

defineProps<{ lawyers: LawyerCard[] }>()
</script>
