<template>
  <!-- fee=null 整卡不渲染：没算过费用时连「费用」这个标题都不该出现 -->
  <section v-if="fee" class="fee-card">
    <h4 class="fee-card__title">费用区间</h4>
    <p v-if="rangeText" class="fee-card__range">{{ rangeText }}</p>
    <p v-else class="fee-card__status">{{ statusText }}</p>
    <!-- 计价基础附在区间旁（≠量级）：丢掉它，「1000~8000 元」会被读成一次性
         收费而不是每小时（③b 真跑 N10 的教训） -->
    <p v-if="fee.charge_basis" class="fee-card__basis">计价基础：{{ fee.charge_basis }}</p>
    <!-- 固定提示必须与区间同屏出现（FR-9.4 / AC-21），文案来自共享常量 -->
    <p class="fee-card__disclaimer">{{ FEE_DISCLAIMER }}</p>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { FeeRange } from '@/core/api/public'
import { FEE_DISCLAIMER } from '@/core/config'

const props = defineProps<{ fee: FeeRange | null }>()

// 区间行：两个端点都在才成立；unit 存在时必须随行渲染（「量级必须带 unit」，
// ③b N10 吃过丢单位的亏）。unit 本身为 null 是后端允许的合法输出（片段里就
// 没有量级字眼）——如实渲染端点、不替后端补单位，也不丢弃一个有效区间
const rangeText = computed(() => {
  const f = props.fee
  if (!f || f.low == null || f.high == null) return ''
  return `${f.low} ~ ${f.high}${f.unit == null ? '' : ` ${f.unit}`}`
})

// 没有区间时给状态文字：no_corpus/rejected 是「没有依据」（设计行为/降级同文案），
// unavailable 是「故障」——两者严格区分是 ③b 红线；不认识的 status 原样透出，
// 不替后端编一句话（文案不编造）
const statusText = computed(() => {
  switch (props.fee?.status) {
    case 'no_corpus':
    case 'rejected':
      return '暂无费用口径依据'
    case 'unavailable':
      return '费用信息暂时不可用'
    default:
      return props.fee?.status ?? ''
  }
})
</script>
