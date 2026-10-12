<!-- 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化 -->
<!-- 问答界面：只做提问、中英文切换、答案与耗时展示、优化前后对比，界面保持清晰简洁。 -->
<template>
  <div class="wrap">
    <!-- 顶部标题与语言切换 -->
    <header>
      <h1>{{ t.title }}</h1>
      <button class="lang" @click="toggleLang">{{ t.lang }}</button>
    </header>

    <!-- 提问区 -->
    <section class="ask">
      <input v-model="question" :placeholder="t.placeholder" @keyup.enter="ask" />
      <button @click="ask" :disabled="loading">{{ loading ? t.running : t.ask }}</button>
      <button class="ghost" @click="compare" :disabled="loading">{{ t.compare }}</button>
    </section>

    <!-- 工单问题清单一键填入，便于逐条演示 -->
    <section class="list">
      <p class="tip">{{ t.pick }}</p>
      <span v-for="q in currentList" :key="q.id" class="chip" @click="question = q.question">
        {{ q.id }}. {{ q.question }}
      </span>
    </section>

    <!-- 优化前后对比结果 -->
    <section v-if="result" class="cards">
      <div class="card" v-if="result.before">
        <h3>{{ t.before }}<span class="badge">PyMuPDF + 固定分块 + 仅向量检索</span></h3>
        <p class="ans">{{ result.before.answer }}</p>
        <p class="meta">{{ t.cost }} {{ result.before.elapsed }} s <em v-if="result.before.error">（{{ result.before.error }}）</em></p>
      </div>
      <div class="card" v-if="result.after || !result.before">
        <h3>
          {{ result.before ? t.after : t.answer }}
          <span class="badge">MinerU + 语义分块 + 混合检索 + 重排</span>
        </h3>
        <p class="ans">{{ current.answer }}</p>
        <p class="meta" :class="{ slow: current.elapsed > 3 }">
          {{ t.cost }} {{ current.elapsed }} s
          <span class="sub">{{ t.retrieval }} {{ current.retrieval_ms }} ms / {{ t.generate }} {{ current.generate_ms }} ms</span>
          <span v-if="current.cached" class="cache">{{ t.cached }}</span>
        </p>
        <!-- 命中片段：展示页码与相关度，便于核对答案出处 -->
        <details v-if="current.contexts && current.contexts.length">
          <summary>{{ t.sources }}</summary>
          <div v-for="(c, i) in current.contexts" :key="i" class="ctx">
            <b>P{{ c.page_no }}</b> （{{ t.score }} {{ c.score }}）{{ c.text }}
          </div>
        </details>
      </div>
    </section>

    <p v-if="error" class="error">{{ error }}</p>
  </div>
</template>

<script setup>
// 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
import { ref, computed } from 'vue'                       // 使用组合式 API

const LANGS = {                                          // 中英文界面文案与对应题库
  zh: { title: '基于 PDF 文档的问答系统（优化版）', lang: 'English', placeholder: '请输入问题，例如：注册资本是多少？',
        ask: '提问', running: '检索生成中…', compare: '优化前后对比', pick: '工单问题清单（点击填入）：',
        before: '优化前', after: '优化后', answer: '答案', cost: '耗时', retrieval: '检索', generate: '生成',
        cached: '缓存命中', sources: '查看命中片段', score: '相关度', empty: '请输入问题' },
  en: { title: 'PDF Document QA System (Optimized)', lang: '中文', placeholder: 'Ask a question, e.g. What is the registered capital?',
        ask: 'Ask', running: 'Retrieving…', compare: 'Compare Before / After', pick: 'Work order questions (click to fill):',
        before: 'Before', after: 'After', answer: 'Answer', cost: 'Latency', retrieval: 'Retrieval', generate: 'Generation',
        cached: 'cache hit', sources: 'Show retrieved passages', score: 'score', empty: 'Please enter a question' }
}
// 工单中列出的 10 个中文检索问题，用于演示逐条提问
const ZH_QUESTIONS = ['报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？','武汉兴图新科电子股份有限公司参与制定了哪个技术标准？','报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？','根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？','武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？','根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？','武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？','武汉兴图新科电子股份有限公司注册资本是多少？','武汉兴图新科电子股份有限公司法定代表人是谁？','武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？']
// 英文问题用于演示多语言问答能力（bge-m3 支持中英跨语言检索）
const EN_QUESTIONS = ['Who is the legal representative of Wuhan Xingtu Xinke Electronics Co., Ltd.?','What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?','In which field has the company become an important supplier?','How much of the raised funds is planned to replenish working capital?']

const lang = ref('zh')                                   // 当前界面语言
const question = ref('')                                 // 当前问题
const loading = ref(false)                               // 请求进行中标记
const result = ref(null)                                 // 结果：{ before, after } 或单答案
const error = ref('')                                    // 错误提示
const t = computed(() => LANGS[lang.value])              // 当前语言文案
const currentList = computed(() => (lang.value === 'zh' ? ZH_QUESTIONS : EN_QUESTIONS).map((q, i) => ({ id: i + 1, question: q })))  // 题库带序号
const current = computed(() => (result.value && (result.value.after || result.value)) || {})  // 当前展示的答案对象

function toggleLang () {                                 // 切换界面语言
  lang.value = lang.value === 'zh' ? 'en' : 'zh'         // 中英互换
  question.value = ''                                    // 清空输入，避免中英混用
  result.value = null                                    // 清空旧结果
}

async function ask () {                                  // 提问：走优化后链路
  if (!question.value.trim()) { error.value = t.value.empty; return }   // 空输入友好提示
  loading.value = true; error.value = ''                                // 置忙并清错误
  try {
    const r = await fetch('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question.value, mode: 'optimized', use_cache: true }) })
    result.value = await r.json()                        // 直接展示后端返回的字段
  } catch (e) { error.value = '请求失败：' + e.message }   // 网络异常提示
  finally { loading.value = false }                      // 复位加载状态
}

async function compare () {                              // 优化前后对比
  if (!question.value.trim()) { error.value = t.value.empty; return }   // 空输入友好提示
  loading.value = true; error.value = ''                                // 置忙并清错误
  try {
    const r = await fetch('/api/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question.value }) })
    result.value = await r.json()                        // { before, after }
  } catch (e) { error.value = '请求失败：' + e.message }   // 网络异常提示
  finally { loading.value = false }                      // 复位加载状态
}
</script>

<style>
/* 简洁样式：单列布局、足够的留白，保证投影演示时清晰可读 */
body { margin: 0; background: #f5f6f8; font-family: "Microsoft YaHei", system-ui, sans-serif; color: #1f2329; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 24px 20px 60px; }
header { display: flex; align-items: center; justify-content: space-between; }
h1 { font-size: 20px; margin: 0; }
.lang { background: #fff; border: 1px solid #d0d3d9; border-radius: 6px; padding: 6px 14px; cursor: pointer; }
.ask { display: flex; gap: 10px; margin: 18px 0 10px; }
.ask input { flex: 1; padding: 12px 14px; border: 1px solid #d0d3d9; border-radius: 8px; font-size: 15px; }
.ask button { padding: 12px 22px; background: #2b6cf6; color: #fff; border: 0; border-radius: 8px; font-size: 15px; cursor: pointer; }
.ask button:disabled { opacity: .6; cursor: default; }
.ask .ghost { background: #fff; color: #2b6cf6; border: 1px solid #2b6cf6; }
.list { margin-bottom: 18px; }
.tip { color: #6b7280; font-size: 13px; margin: 0 0 8px; }
.chip { display: inline-block; background: #fff; border: 1px solid #e3e5e9; border-radius: 20px; padding: 6px 12px; margin: 0 8px 8px 0; font-size: 13px; cursor: pointer; max-width: 100%; }
.chip:hover { border-color: #2b6cf6; color: #2b6cf6; }
.cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 820px) { .cards { grid-template-columns: 1fr; } }
.card { background: #fff; border-radius: 10px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.card h3 { font-size: 15px; margin: 0 0 10px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.badge { font-size: 12px; font-weight: 400; color: #6b7280; background: #f0f2f5; border-radius: 4px; padding: 2px 8px; }
.ans { font-size: 16px; line-height: 1.7; margin: 0 0 10px; white-space: pre-wrap; }
.meta { font-size: 13px; color: #16a34a; margin: 0; }
.meta.slow { color: #dc2626; }
.sub { color: #6b7280; margin-left: 8px; }
.cache { color: #2b6cf6; margin-left: 8px; }
details { margin-top: 10px; font-size: 13px; }
summary { cursor: pointer; color: #2b6cf6; }
.ctx { color: #4b5563; border-top: 1px dashed #e3e5e9; padding: 8px 0; line-height: 1.6; }
.error { color: #dc2626; margin-top: 14px; }
</style>
