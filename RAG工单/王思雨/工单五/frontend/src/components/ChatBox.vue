<!-- 工单编号：人工智能NLP-RAG-Query理解优化 -->
<!-- 文件说明：对话组件。渲染多轮消息、展示引用来源、显示改写后的问题与耗时，并承接输入发送。 -->
<template>
  <div class="wrap">
    <!-- 消息列表：用户气泡与助手气泡左右分列 -->
    <div class="msgs" ref="listEl">
      <div v-for="(m, i) in messages" :key="i" :class="['msg', m.role]">
        <div class="bubble">
          <!-- 助手消息保留换行 -->
          <div class="text" v-html="fmt(m.content)"></div>

          <!-- 助手消息的附加信息：改写结果、耗时、缓存与降级提示 -->
          <div class="meta" v-if="m.role === 'assistant'">
            <span v-if="m.rewritten" class="chip">改写：{{ m.rewritten }}</span><!-- 多轮 Query 理解的可视化 -->
            <span v-if="m.latency" class="chip">{{ m.latency }} ms</span><!-- 响应时间，验收指标之一 -->
            <span v-if="m.cached" class="chip ok">缓存命中</span><!-- 命中缓存 -->
            <span v-for="d in (m.degraded || [])" :key="d" class="chip warn">降级：{{ d }}</span><!-- 降级如实展示 -->
          </div>

          <!-- 引用来源：文件名 + 页码 + 原文片段 -->
          <details class="refs" v-if="m.references && m.references.length">
            <summary>引用来源（{{ m.references.length }} 条）</summary>
            <ol>
              <li v-for="(r, j) in m.references" :key="j">
                <span class="doc">{{ r.doc }} 第 {{ r.page }} 页</span><!-- 出处 -->
                <span class="score">相关度 {{ Number(r.score).toFixed(3) }}</span><!-- 重排得分 -->
                <div class="snippet">{{ r.snippet }}</div><!-- 原文片段 -->
              </li>
            </ol>
          </details>

          <!-- 反馈条：只对助手消息显示 -->
          <FeedbackBar v-if="m.role === 'assistant'" :session-id="sessionId"
                       :question="m.question || ''" :answer="m.content" :rewritten="m.rewritten || ''" />
        </div>
      </div>

      <!-- 等待中的占位提示 -->
      <div v-if="loading" class="msg assistant"><div class="bubble thinking">正在检索并生成…</div></div>
    </div>

    <!-- 输入区：中英文皆可 -->
    <div class="input">
      <textarea v-model="draft" rows="2" placeholder="请输入问题（支持中文 / English），Enter 发送，Shift+Enter 换行"
                @keydown.enter.exact.prevent="submit"></textarea>
      <button :disabled="loading || !draft.trim()" @click="submit">发送</button>
    </div>
    <div class="hint">示例：报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？ ｜ What about Wuhan P&amp;S Information Technology Co., Ltd.?</div>
  </div>
</template>

<script setup>
// 工单编号：人工智能NLP-RAG-Query理解优化
import { nextTick, ref } from 'vue'                                             // 响应式与滚动
import FeedbackBar from './FeedbackBar.vue'                                     // 反馈条子组件

const props = defineProps({ sessionId: { type: String, default: 'default' } })  // 会话号由父组件传入
const messages = ref([])                                                        // 消息列表
const draft = ref('')                                                           // 输入框内容
const loading = ref(false)                                                      // 是否正在请求
const listEl = ref(null)                                                        // 消息容器，用于自动滚动

/** 简单转义并保留换行，避免把模型输出当 HTML 解析。 */
function fmt(s) {
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>')
}

/** 滚到底部，保证最新一轮始终可见。 */
async function scrollDown() {
  await nextTick()                                                              // 等 DOM 更新
  if (listEl.value) listEl.value.scrollTop = listEl.value.scrollHeight          // 滚到底
}

/** 发送一轮问题：写入用户气泡 → 调 /chat → 写入助手气泡（含引用与耗时）。 */
async function submit() {
  const q = draft.value.trim()                                                  // 取问题
  if (!q || loading.value) return                                               // 空或进行中则忽略
  messages.value.push({ role: 'user', content: q })                             // 用户气泡
  draft.value = ''                                                              // 清空输入
  loading.value = true                                                          // 进入加载态
  await scrollDown()                                                            // 滚动
  try {
    const res = await fetch('/api/chat', {                                       // 走 vite 代理
      method: 'POST',                                                           // POST
      headers: { 'Content-Type': 'application/json' },                          // JSON
      body: JSON.stringify({ session_id: props.sessionId, question: q })        // 会话号 + 问题
    })
    const data = await res.json()                                               // 解析
    if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status))         // 非 2xx 抛出可读错误
    messages.value.push({                                                       // 助手气泡
      role: 'assistant',                                                        // 角色
      content: data.answer,                                                     // 回答
      references: data.references || [],                                        // 引用来源
      rewritten: data.rewritten_query || '',                                    // 改写后的问题
      degraded: data.degraded || [],                                            // 降级项
      cached: data.cached,                                                      // 是否命中缓存
      latency: data.latency_ms,                                                 // 端到端耗时
      question: q                                                               // 回带问题，供反馈组件使用
    })
  } catch (e) {
    messages.value.push({ role: 'assistant', content: '请求失败：' + e.message })  // 错误也如实显示在对话里
  } finally {
    loading.value = false                                                       // 退出加载态
    await scrollDown()                                                          // 滚动到底
  }
}

defineExpose({ submit, messages, draft })                                       // 暴露给父组件（预置问题一键发送）
</script>

<style scoped>
.wrap { display: flex; flex-direction: column; height: 100%; min-height: 0; }
.msgs { flex: 1; overflow-y: auto; padding: 14px 16px; background: #f7f9fc; }
.msg { display: flex; margin-bottom: 12px; }
.msg.user { justify-content: flex-end; }
.bubble { max-width: min(760px, 86%); background: #fff; border: 1px solid #e3e8ef; border-radius: 12px; padding: 10px 12px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.msg.user .bubble { background: #e8f0ff; border-color: #c7dcff; }
.text { white-space: normal; line-height: 1.7; color: #1f2937; word-break: break-word; }
.meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.chip { font-size: 12px; background: #eef2f7; color: #475467; border-radius: 999px; padding: 2px 8px; }
.chip.ok { background: #e7f7ec; color: #1b7f3b; }
.chip.warn { background: #fdf1e3; color: #a35b00; }
.refs { margin-top: 8px; font-size: 13px; }
.refs summary { cursor: pointer; color: #1b4fd8; }
.refs ol { margin: 6px 0 0 18px; padding: 0; }
.refs li { margin-bottom: 8px; }
.doc { font-weight: 600; color: #344054; }
.score { margin-left: 8px; color: #667085; font-size: 12px; }
.snippet { color: #475467; background: #f8fafc; border-left: 3px solid #d0d9e6; padding: 6px 8px; margin-top: 4px; border-radius: 4px; white-space: pre-wrap; }
.thinking { color: #667085; }
.input { display: flex; gap: 8px; padding: 10px 14px; border-top: 1px solid #e3e8ef; background: #fff; }
.input textarea { flex: 1; resize: none; border: 1px solid #d5dae2; border-radius: 10px; padding: 8px 10px; font: inherit; }
.input button { border: 0; background: #1b4fd8; color: #fff; border-radius: 10px; padding: 0 18px; cursor: pointer; }
.input button:disabled { background: #b9c4d6; cursor: not-allowed; }
.hint { font-size: 12px; color: #8a94a6; padding: 0 16px 10px; background: #fff; }
</style>
