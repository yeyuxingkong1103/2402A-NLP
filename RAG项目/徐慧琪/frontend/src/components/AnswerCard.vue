<template>
  <article class="answer-card">
    <!-- 单轮系统的如实提示：后端不收会话，补充信息要作为新的一次提问 -->
    <p v-if="result.status === 'need_more_info'" class="answer-card__status">
      补充后可再问（每次提问独立）
    </p>
    <section class="answer-card__answer">
      <h4 class="answer-card__label">AI 解读，仅供参考</h4>
      <!-- 正文只走文本插值（不 v-html）：模型输出里的 <b>…</b> 会原样显示为
           字面量而不是变成 DOM（XSS 防线，设计 §六） -->
      <p class="answer-card__body" style="white-space: pre-wrap">{{ result.answer }}</p>
    </section>
    <section class="answer-card__citations">
      <h4 class="answer-card__heading">法条原文</h4>
      <CitationList :citations="result.citations" />
    </section>
    <!-- 参考块（低强调）：命中的检索落点，供人工核对，不参与结论呈现 -->
    <section v-if="result.sources.length" class="answer-card__sources">
      <h4 class="answer-card__heading">参考</h4>
      <ul class="answer-card__source-list">
        <li v-for="(s, i) in result.sources" :key="i" class="answer-card__source">
          <small>第{{ s.article_no }}条 · {{ s.path }}</small>
        </li>
      </ul>
    </section>
    <!-- 律师卡与费用卡只有公众响应才渲染：按**键在不在**判形状，而不是按值
         空不空——律师侧的 7 键里根本没有这两个键（schemas.py 基类/子类分工） -->
    <template v-if="publicResult">
      <LawyerCards :lawyers="publicResult.lawyers" />
      <FeeCard :fee="publicResult.fee_range" />
    </template>
    <!-- 免责声明来自 API 字段：空串不渲染，非空逐字透出（不编造） -->
    <p v-if="result.disclaimer" class="answer-card__disclaimer">{{ result.disclaimer }}</p>
    <el-button v-if="copyable" class="answer-card__copy" size="small" @click="onCopy">
      复制
    </el-button>
  </article>
</template>

<script setup lang="ts">
import { computed } from 'vue'
// 裁决 1：共享类型显式 import type——编译期擦除，公众入口的模块图里因此不存在
// lawyer.ts（结构不变式是**单向**的：反向不成立——律师入口读 public.ts 的
// nav/article 是允许的，Task 6 审查裁定）；Task 7 的 check:bundle 扫运行时代码为证
import type { PublicQAAnswer } from '@/core/api/public'
import type { QAAnswer } from '@/core/api/lawyer'
import { buildCopyText } from '@/core/format'
import CitationList from '@/components/CitationList.vue'
import FeeCard from '@/components/FeeCard.vue'
import LawyerCards from '@/components/LawyerCards.vue'

const props = defineProps<{ result: QAAnswer | PublicQAAnswer; copyable?: boolean }>()
const emit = defineEmits<{ copied: [text: string] }>()

// 运行时形状判定：public 响应恒带这两键（值可为 [] / null），lawyer 响应里
// 这两键不存在——「键的有无是契约」（schemas.py 的注释）
const publicResult = computed((): PublicQAAnswer | null =>
  'lawyers' in props.result ? props.result as PublicQAAnswer : null)

// 复制文本由组件按 FR-7.4 格式化后随事件带出；写剪贴板是页面副作用（裁决 2）
function onCopy() {
  emit('copied', buildCopyText(props.result.answer, props.result.citations))
}
</script>
