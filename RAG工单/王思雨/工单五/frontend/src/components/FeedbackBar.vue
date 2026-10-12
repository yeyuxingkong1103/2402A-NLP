<!-- 工单编号：人工智能NLP-RAG-Query理解优化 -->
<!-- 文件说明：反馈条组件。每条回答下方提供 点赞/点踩/文字意见，满足工单"支持多轮对话和用户反馈"的要求。 -->
<template>
  <div class="fb">
    <!-- 点赞/点踩按钮：已选中时高亮，避免重复提交同一评价 -->
    <button :class="{ on: local === 1 }" @click="send(1)" title="回答有帮助">👍 有帮助</button>
    <button :class="{ on: local === -1 }" @click="send(-1)" title="回答有问题">👎 有问题</button>
    <!-- 展开文字意见输入框 -->
    <button class="ghost" @click="open = !open">{{ open ? '收起' : '文字反馈' }}</button>
    <!-- 提交后的提示文字 -->
    <span class="tip" v-if="tip">{{ tip }}</span>
    <!-- 文字意见区域：输入 + 提交 -->
    <div class="row" v-if="open">
      <input v-model="comment" placeholder="请描述问题，例如：引用了错误的公司 / 数字与原文不符" @keyup.enter="send(local || 0)" />
      <button @click="send(local || 0)">提交</button>
    </div>
  </div>
</template>

<script setup>
// 工单编号：人工智能NLP-RAG-Query理解优化
import { ref } from 'vue'                                                       // 响应式引用

// 父组件传入的上下文：会话号、本轮问题、本轮回答、本轮改写后的问题
const props = defineProps({
  sessionId: { type: String, default: 'default' },                              // 会话号
  question: { type: String, default: '' },                                      // 被评价的问题
  answer: { type: String, default: '' },                                        // 被评价的回答
  rewritten: { type: String, default: '' }                                      // 改写后的问题，便于定位问题
})

const local = ref(0)                                                            // 本地评分：1 赞 / -1 踩 / 0 中立
const comment = ref('')                                                         // 文字意见
const open = ref(false)                                                         // 是否展开输入框
const tip = ref('')                                                             // 提交提示

/** 提交反馈到 /feedback；失败时如实提示，不假装成功。 */
async function send(rating) {
  local.value = rating                                                          // 记录本次评分（点踩按钮也走这里）
  try {
    const res = await fetch('/api/feedback', {                                  // 走 vite 代理到后端
      method: 'POST',                                                           // 提交
      headers: { 'Content-Type': 'application/json' },                          // JSON 体
      body: JSON.stringify({                                                    // 反馈内容
        session_id: props.sessionId,                                            // 会话号
        question: props.question,                                               // 问题
        answer: props.answer,                                                   // 回答
        rating: rating,                                                         // 评分
        comment: comment.value,                                                 // 文字意见
        rewritten_query: props.rewritten                                        // 改写后的问题
      })
    })
    const data = await res.json()                                               // 解析响应
    tip.value = data.message || (data.ok ? '反馈已记录' : '反馈失败')               // 展示后端返回的提示
    if (data.ok) { comment.value = ''; open.value = false }                     // 成功则收起输入框
  } catch (e) {
    tip.value = '反馈提交失败：' + e.message                                      // 网络异常如实提示
  }
  setTimeout(() => { tip.value = '' }, 3000)                                    // 3 秒后淡出提示
}
</script>

<style scoped>
.fb { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 6px; font-size: 12px; }
button { border: 1px solid #d5dae2; background: #fff; border-radius: 999px; padding: 3px 10px; cursor: pointer; color: #46505f; }
button.on { background: #e8f0ff; border-color: #7aa7ff; color: #1b4fd8; }      /* 选中态高亮 */
button.ghost { background: transparent; border-style: dashed; }
.tip { color: #1b7f3b; }
.row { display: flex; gap: 6px; width: 100%; margin-top: 4px; }
.row input { flex: 1; border: 1px solid #d5dae2; border-radius: 8px; padding: 5px 8px; }
</style>
