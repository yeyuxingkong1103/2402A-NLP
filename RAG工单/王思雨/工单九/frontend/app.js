// 工单编号：人工智能NLP-RAG-Graph RAG优化任务 —— 前端逻辑（Vue 3，Options API）
// 只做界面与请求编排，所有算法在后端；不含任何评测数字的硬编码，指标一律从 eval/*.json 读
Vue.createApp({
  data() {                                                    // 组件状态
    return {
      tab: 'ask',                                             // 当前标签页
      q: '', mode: 'after', useCache: true, busy: false, ans: null,   // 问答区
      graphData: { nodes: [], edges: [], communities: [], stats: null }, // 图谱区
      bf: null, af: null, evalHint: '（尚未读取）',             // 指标对比区
      file: null, upMsg: '', rawText: '', rawCode: '', txMsg: '', healthSt: '' // 上传/健康
    }
  },
  computed: {
    evalRows() {                                              // 把 before/after JSON 摊平成表格行
      const g = (o, k) => (o && o.ragas && o.ragas[k] != null) ? o.ragas[k] : null   // 取指标值
      const mk = (name, key, target, num) => {                // 构造一行
        const b = num ? (this.bf || {})[key] : g(this.bf, key)
        const a = num ? (this.af || {})[key] : g(this.af, key)
        return { name, target,                                     // 名称 / 目标线
                 b: b != null ? b : '待填充', a: a != null ? a : '待填充',   // 无数据显示“待填充”，不显示 ❌ 以免被误读成未达标
                 d: (b != null && a != null) ? (num ? (a - b) + ' ms' : (a - b).toFixed(4)) : '—',  // 增量
                 ok: a != null && (num ? a <= target : a >= target),    // 达标判定：无数据为 false
                 has: a != null }                                       // 是否有数据（决定显示 ✅/❌ 还是 —）
      }
      return [ mk('context_precision', 'llm_context_precision_with_reference', 0.8),  // 工单验收：≥0.8
               mk('context_recall', 'llm_context_recall', 0.9),                       // 工单验收：≥0.9
               mk('时延 p50(ms)', 'p50_ms', 3000, true), mk('时延 max(ms)', 'max_ms', 3000, true) ] // 性能验收：≤3000ms
    }
  },
  methods: {
    async post(url, body) {                                   // 统一 POST，错误转成可读文本
      const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const d = await r.json(); if (!r.ok) throw new Error(d.detail || r.status); return d
    },
    async ask() {                                             // 提问：按 mode 调 /ask 或 /ask/before
      this.busy = true; this.ans = null                        // 置忙、清旧答案
      try { this.ans = await this.post(this.mode === 'after' ? '/ask' : '/ask/before', { question: this.q, use_cache: this.useCache, lang: 'auto' }) }
      catch (e) { this.ans = { answer: '请求失败：' + e.message, contexts: [], graph: {}, mode: this.mode, latency_ms: 0, llm_ms: 0 } }  // 失败也要给可读反馈
      this.busy = false                                        // 解除忙
    },
    async loadGraph() {                                        // 拉图谱主干与社区摘要
      const d = await (await fetch('/graph?limit=150')).json(); this.graphData = d
    },
    async loadEval() {                                         // 拉两份评估结果并给提示
      const b = await (await fetch('/results/before')).json(); const a = await (await fetch('/results/after')).json()
      this.bf = b.empty ? null : b; this.af = a.empty ? null : a   // 缺文件时置 null
      this.evalHint = (b.empty || a.empty) ? ('缺文件：' + ((b.hint || '') + ' ' + (a.hint || ''))) : '已读取 eval/before.json 与 eval/after.json'  // 明确提示怎么生成
    },
    onFile(e) { this.file = e.target.files[0] || null },        // 记录所选文件
    async upload() {                                            // 上传 PDF：FormData 直传
      this.busy = true; this.upMsg = '上传中…'                   // 置忙
      const fd = new FormData(); fd.append('file', this.file)     // 组装表单
      try {
        const r = await fetch('/upload', { method: 'POST', body: fd }); const d = await r.json()   // 提交
        this.upMsg = r.ok ? `解析器=${d.parser} 字符数=${d.chars} 建图后=${JSON.stringify(d.graph)}` : '失败：' + (d.detail || r.status)  // 成功/失败都给可读信息
      } catch (e) { this.upMsg = '失败：' + e.message }           // 网络异常
      this.busy = false                                          // 解除忙
    },
    async buildText() {                                          // 贴文本建图
      try { this.txMsg = '已并入图谱 → ' + JSON.stringify(await this.post('/graph/text', { text: this.rawText, code: this.rawCode })) }
      catch (e) { this.txMsg = '失败：' + e.message }              // 明确报错
    },
    async health() {                                             // 健康检查（Milvus/Redis/图谱/Key）
      try { this.healthSt = JSON.stringify(await (await fetch('/health')).json()) } catch (e) { this.healthSt = '失败：' + e.message }
    }
  },
  mounted() { this.loadGraph(); this.health() }                  // 首屏自动拉图谱与状态
}).mount('#app')
