<!-- 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化 -->
<!-- 文件说明：问答界面。输入问题（中/英）→ 显示答案、来源片段、页码与表格定位、检索精确度与耗时。 -->
<script setup>
import { onMounted, ref } from 'vue'                                    // Vue 组合式 API

const API = '/api'                                                       // 后端接口前缀（由 vite 代理到 8000）
const question = ref('')                                                 // 问题输入
const lang = ref('auto')                                                 // 语言：auto / zh / en
const loading = ref(false)                                               // 请求中标记
const result = ref(null)                                                 // 问答结果
const error = ref('')                                                    // 错误提示
const health = ref(null)                                                 // 依赖健康状态
const samples = ref([])                                                  // 工单 14 题，供一键填入

// 工单指定的 4 道力源信息题与其余兴图新科题，逐条陈列便于演示
const EXAMPLES = [
  '武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？',
  '武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？',
  '与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？',
  '与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？',
  '报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？',
  '武汉兴图新科电子股份有限公司参与制定了哪个技术标准？',
  '报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？',
  '根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？',
  '武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？',
  '根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？',
  '武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？',
  '武汉兴图新科电子股份有限公司注册资本是多少？',
  '武汉兴图新科电子股份有限公司法定代表人是谁？',
  '武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？'
]
const EN_EXAMPLE = 'Who is the legal representative of Wuhan Xingtu Xinke Electronics Co., Ltd.?'  // 英文提问示例

// 提交提问。用 SSE 流式：检索一完成（约 0.9 秒）就能先渲染来源片段与表格定位，
// 不必等大模型生成完——用户感知的等待从"整个端到端"降到"检索完成"。
// 同一个 /ask 接口，只是带 Accept: text/event-stream；后端不带该头时仍返回一次性 JSON。
async function ask() {
  const q = question.value.trim()                                         // 取出去空白后的问题
  if (!q) { error.value = '请输入需要查询的问题。'; return }                 // 前端兜底校验
  loading.value = true; error.value = ''; result.value = null             // 进入加载态
  const draft = { answer: '', contexts: [], streaming: true }             // 边收边渲染的草稿对象
  try {
    const res = await fetch(`${API}/ask`, {                               // POST 提问
      method: 'POST',                                                      // 提问是写操作
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },  // 声明要流式
      body: JSON.stringify({ question: q, lang: lang.value })              // 问题与语言
    })
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)         // 非 200 直接抛
    const reader = res.body.getReader()                                    // 流读取器
    const decoder = new TextDecoder()                                      // 按 UTF-8 增量解码（中文是多字节，不能逐字节解）
    let buf = ''                                                           // 跨 chunk 的半帧缓冲
    for (;;) {                                                             // 逐块读取
      const { value, done } = await reader.read()                          // 取一块
      if (done) break                                                      // 流结束
      buf += decoder.decode(value, { stream: true })                       // 增量解码
      const frames = buf.split('\n\n'); buf = frames.pop()                 // 按空行切帧，最后一段可能不完整
      for (const f of frames) {                                            // 逐帧处理
        const ev = (f.match(/^event: (.+)$/m) || [])[1]                    // 事件名
        const raw = (f.match(/^data: (.+)$/m) || [])[1]                    // 数据行
        if (!ev || !raw) continue                                          // 空帧跳过
        const d = JSON.parse(raw)                                          // 解析载荷
        if (ev === 'retrieval') {                                          // 检索完成：先渲染来源与指标
          Object.assign(draft, d); result.value = { ...draft }              // 答案此时还是空的
        } else if (ev === 'token') {                                       // 逐段追加答案
          draft.answer += d.text; result.value = { ...draft }               // 触发视图更新
        } else if (ev === 'reset') {                                       // 生成失败，清掉已推的半截内容
          draft.answer = ''; draft.degraded = true; draft.degraded_reason = d.reason
          result.value = { ...draft }
        } else if (ev === 'done') {                                        // 收尾：以完整结果为准
          result.value = { ...d }
        } else if (ev === 'error') {                                       // 后端明确报错
          error.value = d.detail || d.answer || '请求失败'
          if (d.answer) result.value = { ...draft, ...d }
        }
      }
    }
  } catch (e) { error.value = '无法连接后端服务，请确认 FastAPI 已启动（8000 端口）。' }
  finally { loading.value = false; if (result.value) delete result.value.streaming }  // 退出加载态
}

async function loadHealth() {                                              // 读取依赖健康状态
  try { health.value = await (await fetch(`${API}/health`)).json() }        // 后端探活
  catch { health.value = { api: 'error' } }                                // 后端未启动时不阻塞界面
}

function fill(q) { question.value = q; ask() }                            // 点示例即提问
onMounted(loadHealth)                                                      // 首屏加载健康状态
</script>

<template>
  <div class="wrap">
    <header>
      <h1>招股说明书表格解析与检索问答</h1>
      <p class="sub">
        工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化 ·
        知识库：招股说明书1（兴图新科） / 招股说明书2（力源信息）
      </p>
      <!-- 依赖状态：任一依赖异常也照常展示，便于排障 -->
      <div class="health" v-if="health">
        <span :class="health.api === 'ok' ? 'ok' : 'bad'">FastAPI {{ health.api }}</span>
        <span :class="health.milvus === 'ok' ? 'ok' : 'bad'">Milvus {{ health.milvus }}</span>
        <span :class="health.redis === 'ok' ? 'ok' : 'bad'">Redis {{ health.redis }}</span>
        <span :class="health.llm_key ? 'ok' : 'bad'">DeepSeek Key {{ health.llm_key ? '已配置' : '缺失' }}</span>
        <span :class="health.index ? 'ok' : 'bad'">索引 {{ health.index ? '已构建' : '未构建' }}</span>
      </div>
    </header>

    <section class="ask">
      <textarea v-model="question" rows="2" placeholder="请输入问题，支持中文或英文…" @keydown.ctrl.enter="ask"></textarea>
      <div class="row">
        <select v-model="lang">
          <option value="auto">自动识别语言</option>
          <option value="zh">中文问答</option>
          <option value="en">English</option>
        </select>
        <button :disabled="loading" @click="ask">{{ loading ? '检索中…' : '提问' }}</button>
      </div>
      <div class="examples">
        <span class="tag" @click="fill(EN_EXAMPLE)">EN 示例</span>
        <span class="tag" v-for="q in EXAMPLES" :key="q" @click="fill(q)">{{ q.slice(0, 18) }}…</span>
      </div>
    </section>

    <p class="error" v-if="error">{{ error }}</p>

    <section class="answer" v-if="result">
      <h2>答案</h2>
      <!-- 降级提示：Milvus / Redis / 大模型任一不可用时都必须让用户看见，不能静默返回低质量结果 -->
      <p class="degraded" v-if="result.degraded">⚠️ {{ result.degraded_reason }}</p>
      <p class="text">{{ result.answer }}</p>

      <!-- 检索精确度与分阶段耗时。缓存命中时端到端是 0s，而分段耗时是首次计算时留下的，
           必须标明，否则会被误读成"这次请求花了两秒多检索"。 -->
      <div class="metrics">
        <div><b>{{ result.elapsed }}</b><span>端到端耗时 (s)</span></div>
        <div><b>{{ result.retrieval_ms ?? '—' }}</b>
             <span>检索+重排 (ms){{ result.cached ? ' · 首次计算' : '' }}</span></div>
        <div><b>{{ result.generate_ms ?? '—' }}</b>
             <span>生成 (ms){{ result.cached ? ' · 首次计算' : '' }}</span></div>
        <div><b>{{ result.retrieval_mode || '—' }}</b><span>检索通道</span></div>
        <div><b>{{ result.cached ? '命中' : '未命中' }}</b><span>Redis 缓存</span></div>
        <div><b>{{ (result.contexts || []).length }}</b><span>来源片段数</span></div>
      </div>

      <h3>来源片段 · 页码 · 表格定位</h3>
      <div class="src" v-for="(c, i) in result.contexts" :key="i">
        <div class="head">
          <span class="idx">#{{ i + 1 }}</span>
          <span class="doc">{{ c.doc }}</span>
          <span class="page">P{{ c.page_no }}<template v-if="c.page_end !== c.page_no">–{{ c.page_end }}</template></span>
          <span class="loc" v-if="c.locator">{{ c.locator }}</span>
          <span class="cap" v-if="c.table_caption">表题：{{ c.table_caption }}</span>
          <span class="type" :class="c.block_type">{{ c.block_type }}</span>
          <span class="score">相关度 {{ c.score }}</span>
        </div>
        <pre>{{ c.text }}</pre>
      </div>
    </section>
  </div>
</template>

<style>
/* 保持简洁：单栏布局，来源片段按定位信息分行展示 */
* { box-sizing: border-box; }
body { margin: 0; background: #f5f6f8; color: #1f2328;
       font: 14px/1.65 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
.wrap { max-width: 960px; margin: 0 auto; padding: 24px 18px 60px; }
h1 { font-size: 21px; margin: 0 0 6px; }
.sub { color: #6b7280; margin: 0 0 10px; font-size: 12.5px; }
.health { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.health span { font-size: 12px; padding: 2px 8px; border-radius: 10px; background: #e8eaed; }
.health .ok { background: #e6f4ea; color: #137333; }
.health .bad { background: #fce8e6; color: #c5221f; }
.ask, .answer { background: #fff; border: 1px solid #e3e5e8; border-radius: 10px;
                padding: 16px; margin-bottom: 16px; }
textarea { width: 100%; padding: 10px; border: 1px solid #d8dade; border-radius: 8px;
           font: inherit; resize: vertical; }
.row { display: flex; gap: 10px; margin-top: 10px; align-items: center; }
select, button { font: inherit; padding: 7px 14px; border-radius: 8px; border: 1px solid #d8dade; background: #fff; }
button { background: #1a73e8; color: #fff; border-color: #1a73e8; cursor: pointer; }
button:disabled { opacity: .6; cursor: default; }
.examples { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.tag { font-size: 12px; padding: 3px 9px; background: #eef2f7; border-radius: 10px;
       cursor: pointer; color: #35506b; }
.tag:hover { background: #dde7f3; }
.error { color: #c5221f; background: #fce8e6; padding: 10px 14px; border-radius: 8px; }
.degraded { color: #b06000; background: #fff4e5; border-left: 3px solid #f0a020;
            padding: 8px 12px; border-radius: 6px; margin: 0 0 10px; font-size: 13px; }
.answer h2 { font-size: 16px; margin: 0 0 8px; }
.answer h3 { font-size: 14px; margin: 20px 0 8px; color: #374151; }
.text { white-space: pre-wrap; margin: 0; padding: 12px; background: #f8f9fa; border-radius: 8px; }
.metrics { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 12px; }
.metrics div { display: flex; flex-direction: column; }
.metrics b { font-size: 16px; color: #1a73e8; }
.metrics span { font-size: 11.5px; color: #6b7280; }
.src { border: 1px solid #e8eaed; border-radius: 8px; padding: 10px; margin-bottom: 10px; }
.src .head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 12px; margin-bottom: 6px; }
.idx { color: #9aa0a6; }
.doc { background: #eef2f7; padding: 1px 7px; border-radius: 8px; }
.page { background: #e6f4ea; color: #137333; padding: 1px 7px; border-radius: 8px; }
.loc { background: #fff4e5; color: #b06000; padding: 1px 7px; border-radius: 8px; }
.cap { color: #6b7280; }
.type { padding: 1px 7px; border-radius: 8px; background: #eceff1; }
.type.table_row, .type.table { background: #fde7f3; color: #a3257b; }
.score { margin-left: auto; color: #1a73e8; font-weight: 600; }
pre { margin: 0; white-space: pre-wrap; word-break: break-all; font: 12.5px/1.6 inherit;
      background: #f8f9fa; padding: 8px; border-radius: 6px; }
</style>
