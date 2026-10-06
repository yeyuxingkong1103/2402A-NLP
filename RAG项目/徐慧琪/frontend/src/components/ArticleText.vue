<template>
  <section class="article-text">
    <header class="article-text__head">
      <!-- 中文条号优先（律师写文书用「第五百八十四条」）；后端可能给 null，
           此时退回阿拉伯条号而不是渲染一个空标题 -->
      <h3 class="article-text__no">{{ article.article_no_cn ?? `第${article.article_no}条` }}</h3>
      <!-- 效力徽标：引用该条前要据此判断是否现行有效（status 来自 law_version） -->
      <el-tag class="article-text__status" size="small"
        :type="article.status === '现行有效' ? 'success' : 'info'">{{ article.status }}</el-tag>
      <p class="article-text__path">{{ article.path }}</p>
    </header>
    <!-- 纯文本 + pre-wrap：换行与空格原样保留，且不解析任何标记（XSS 防线） -->
    <p class="article-text__body" style="white-space: pre-wrap">{{ article.text }}</p>
    <el-button v-if="copyable" class="article-text__copy" size="small" @click="onCopy">
      复制引用
    </el-button>
  </section>
</template>

<script setup lang="ts">
import type { ArticleResponse } from '@/core/api/public'
import { formatCitation } from '@/core/format'

const props = defineProps<{ article: ArticleResponse; copyable?: boolean }>()
const emit = defineEmits<{ copied: [text: string] }>()

// 复制的是 FR-7.4 的规范引用（带条款号），不是裸正文：条文详情这一层没有
// 款/项，quote 取全文；剪贴板写入由页面做（组件只 emit，裁决 2）
function onCopy() {
  emit('copied', formatCitation({
    article: props.article.article_no, paragraph: null, item: null, quote: props.article.text,
  }))
}
</script>
