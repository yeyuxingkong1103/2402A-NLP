<!-- 工单编号：人工智能NLP-RAG-Query理解优化 -->
<!-- 文件说明：根组件。左侧展示依赖健康状态、知识库与工单五验收问题；右侧为多轮对话区。 -->
<template>
  <div class="app">
    <!-- 顶栏：标题 + 会话号 + 健康状态 -->
    <header>
      <div class="title">招股说明书多轮问答 <span class="sub">Query 理解优化（工单五）</span></div>
      <div class="right">
        <span class="sid">会话：{{ sessionId }}</span><!-- 便于多会话隔离演示 -->
        <button class="ghost" @click="resetSession">新会话</button><!-- 清空多轮上下文与实体 -->
        <span :class="['dot', health.status === 'ok' ? 'ok' : 'warn']"></span><!-- 健康指示灯 -->
        <span class="hstat">{{ health.status === 'ok' ? '依赖正常' : '降级运行' }}</span>
      </div>
    </header>

    <div class="body">
      <!-- 侧栏 -->
      <aside>
        <section>
          <h4>依赖状态</h4>
          <ul class="deps">
            <li><b>知识库</b><span>{{ health.chunks }} 块 / {{ (health.docs || []).length }} 篇</span></li>
            <li><b>Milvus</b><span :class="yn(health.milvus)">{{ health.milvus ? '可用' : '不可用（仅关键词检索）' }}</span></li>
            <li><b>Redis</b><span :class="yn(health.redis)">{{ health.redis ? '可用' : '不可用（无历史/缓存）' }}</span></li>
            <li><b>DeepSeek</b><span :class="yn(health.llm)">{{ health.llm ? '已配置' : '未配置' }}</span></li>
            <li><b>向量模型</b><span :class="yn(health.embed_model)">{{ health.embed_model ? '已加载' : '未加载' }}</span></li>
            <li><b>重排模型</b><span :class="yn(health.rerank_model)">{{ health.rerank_model ? '已加载' : '降级为原始排序' }}</span></li>
          </ul>
          <p class="detail" v-if="health.detail">{{ health.detail }}</p><!-- 降级说明 -->
        </section>

        <section>
          <h4>知识库文档</h4>
          <ul class="docs"><li v-for="d in health.docs || []" :key="d">{{ d }}</li></ul>
        </section>

        <section>
          <h4>工单验收多轮问题</h4>
          <ol class="qs">
            <li v-for="(q, i) in presets" :key="i" @click="ask(q)" :title="q">{{ q }}</li>
          </ol>
          <p class="tip">按顺序点击即可复现工单五的多轮对话（Q4 依赖 Q3 的实体切换）。</p>
        </section>

        <section>
          <h4>英文提问示例</h4>
          <ol class="qs en">
            <li v-for="(q, i) in enPresets" :key="i" @click="ask(q)" :title="q">{{ q }}</li>
          </ol>
        </section>
      </aside>

      <!-- 对话主区 -->
      <main>
        <ChatBox ref="chat" :session-id="sessionId" />
      </main>
    </div>
  </div>
</template>

<script setup>
// 工单编号：人工智能NLP-RAG-Query理解优化
import { onMounted, reactive, ref } from 'vue'                                  // 响应式 API
import ChatBox from './components/ChatBox.vue'                                  // 对话组件

const chat = ref(null)                                                          // 对话组件引用
const sessionId = ref('demo-' + Math.random().toString(36).slice(2, 8))         // 会话号：随机生成，保证多会话隔离
const health = reactive({ status: 'unknown', chunks: 0, docs: [],               // 健康状态初始值
                          milvus: false, redis: false, llm: false,              // 依赖项
                          embed_model: false, rerank_model: false, detail: '' })  // 模型与说明

// 工单五验收的 5 个多轮问题（原文照抄，便于验收时逐条点击）
const presets = [
  '报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？',
  '他参与的哪个工程荣获了国家科技进步一等奖？',
  '这个公司的法定代表人是谁？',
  '那武汉力源信息技术股份有限公司呢？',
  '武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？'
]
// 英文提问示例（工单要求支持中英文问答）
const enPresets = [
  'What are the military-related revenues of Wuhan Xingtu Xinke Electronics Co., Ltd. during the reporting period?',
  'Who is the legal representative of this company?',
  'What about Wuhan P&S Information Technology Co., Ltd.?'
]

/** 把布尔值映射成颜色类名。 */
function yn(v) { return v ? 'ok' : 'bad' }

/** 拉取后端健康状态，用于侧栏展示与首屏提示。 */
async function loadHealth() {
  try {
    const r = await fetch('/api/health')                                        // 走 vite 代理
    Object.assign(health, await r.json())                                       // 合并进响应式对象
  } catch (e) {
    health.status = 'down'; health.detail = '无法连接后端：' + e.message          // 后端未启动时如实提示
  }
}

/** 点击预置问题：写入输入框并直接发送，保证多轮上下文连续。 */
function ask(q) {
  if (!chat.value) return                                                       // 组件未就绪
  chat.value.draft = q                                                          // 填进输入框
  chat.value.submit()                                                           // 触发发送
}

/** 开新会话：换会话号并清空界面，避免上一轮的实体被继承。 */
function resetSession() {
  sessionId.value = 'demo-' + Math.random().toString(36).slice(2, 8)            // 新会话号
  if (chat.value) chat.value.messages = []                                      // 清空消息
}

onMounted(() => { loadHealth(); setInterval(loadHealth, 20000) })               // 首屏拉一次，之后每 20 秒刷新
</script>

<style>
/* 全局基础样式 */
* { box-sizing: border-box; }
html, body, #app { height: 100%; margin: 0; }
body { font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif; color: #1f2937; }
.app { display: flex; flex-direction: column; height: 100%; }
header { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; background: #0f1b33; color: #fff; }
.title { font-size: 16px; font-weight: 600; }
.sub { font-size: 12px; opacity: .7; margin-left: 8px; }
header .right { display: flex; align-items: center; gap: 10px; font-size: 12px; }
header .sid { opacity: .8; }
header button.ghost { background: transparent; color: #cfe0ff; border: 1px solid #39527f; border-radius: 8px; padding: 3px 10px; cursor: pointer; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot.ok { background: #37d67a; }
.dot.warn { background: #ffb020; }
.body { flex: 1; display: flex; min-height: 0; }
aside { width: 330px; border-right: 1px solid #e3e8ef; padding: 12px 14px; overflow-y: auto; background: #fff; }
aside h4 { margin: 12px 0 6px; font-size: 13px; color: #344054; }
aside ul, aside ol { margin: 0; padding-left: 18px; font-size: 12.5px; }
aside li { margin-bottom: 4px; }
.deps li { display: flex; justify-content: space-between; gap: 8px; }
.deps .ok { color: #1b7f3b; }
.deps .bad { color: #b23b3b; }
.qs { cursor: pointer; color: #1b4fd8; }
.qs li { line-height: 1.5; margin-bottom: 6px; }
.qs li:hover { text-decoration: underline; }
.detail { font-size: 12px; color: #a35b00; background: #fdf6ec; border-radius: 6px; padding: 6px 8px; }
.tip { font-size: 12px; color: #8a94a6; margin: 6px 0 0; }
main { flex: 1; min-width: 0; display: flex; flex-direction: column; }
</style>
