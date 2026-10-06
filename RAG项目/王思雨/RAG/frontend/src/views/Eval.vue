<!-- 第 10 步：评测结果页面，展示 RAGAS 四项指标汇总与逐条明细 -->
<template>
  <!-- 页面容器 -->
  <div class="page">
    <!-- 汇总卡片 -->
    <el-card>
      <!-- 标题带刷新按钮 -->
      <template #header>
        <span>RAGAS 评测汇总</span>
        <el-button link type="primary" class="right" @click="load">刷新</el-button>
      </template>

      <!-- 评估模型、向量模型与评测时间 -->
      <el-descriptions :column="3" border size="small">
        <!-- 评估用大模型 -->
        <el-descriptions-item label="评估模型">{{ data.model || '-' }}</el-descriptions-item>
        <!-- 评估用向量模型 -->
        <el-descriptions-item label="评估 Embedding">{{ data.embedding || '-' }}</el-descriptions-item>
        <!-- 评测时间：时间戳转成可读文本 -->
        <el-descriptions-item label="评测时间">{{ timeText }}</el-descriptions-item>
      </el-descriptions>

      <!-- 四个指标：横向平铺 -->
      <div class="metrics">
        <!-- 逐个指标渲染一个方块 -->
        <div v-for="m in METRICS" :key="m.key" class="metric">
          <!-- 指标中文名 -->
          <div class="name">{{ m.label }}</div>
          <!-- 指标均分，保留 4 位小数 -->
          <div class="value">{{ fmt(data.summary[m.key]) }}</div>
        </div>
      </div>
    </el-card>

    <!-- 逐条明细卡片 -->
    <el-card class="gap">
      <!-- 标题显示条数 -->
      <template #header>逐条明细（{{ rows.length }} 条）</template>
      <!-- 有数据时展示表格 -->
      <el-table v-if="rows.length" :data="rows" size="small" border>
        <!-- 评测问题 -->
        <el-table-column prop="question" label="问题" min-width="200" show-overflow-tooltip />
        <!-- 系统回答：过长自动截断，鼠标悬浮展开全文 -->
        <el-table-column prop="answer" label="系统回答" min-width="260" show-overflow-tooltip />
        <!-- 参考答案 -->
        <el-table-column prop="ground_truth" label="参考答案" min-width="220" show-overflow-tooltip />
        <!-- 忠实度 -->
        <el-table-column label="忠实度" width="90">
          <template #default="{ row }">{{ fmt(row.faithfulness) }}</template>
        </el-table-column>
        <!-- 答案相关度 -->
        <el-table-column label="答案相关度" width="110">
          <template #default="{ row }">{{ fmt(row.answer_relevancy) }}</template>
        </el-table-column>
        <!-- 上下文精确率 -->
        <el-table-column label="上下文精确率" width="120">
          <template #default="{ row }">{{ fmt(row.context_precision) }}</template>
        </el-table-column>
        <!-- 上下文召回率 -->
        <el-table-column label="上下文召回率" width="120">
          <template #default="{ row }">{{ fmt(row.context_recall) }}</template>
        </el-table-column>
      </el-table>

      <!-- 无数据时的占位提示 -->
      <el-empty v-else description="暂无评测数据，请先运行 python -m eval_ragas" />
    </el-card>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive } from 'vue'   // 导入响应式工具与挂载钩子
import { evalResult } from '../api'                  // 导入评测结果接口

// 四个指标的键与中文名，模板与表格共用同一份定义
const METRICS = [
  { key: 'faithfulness', label: '忠实度' },           // 回答是否忠于检索到的上下文
  { key: 'answer_relevancy', label: '答案相关度' },     // 回答是否切题
  { key: 'context_precision', label: '上下文精确率' },  // 检索结果是否精准
  { key: 'context_recall', label: '上下文召回率' }      // 依据是否被召回全
]

// 接口返回的整体结构；先给默认值，保证加载前模板不报错
const data = reactive({ timestamp: 0, model: '', embedding: '', summary: {}, details: [] })
const rows = computed(() => data.details || [])       // 明细表格数据源

// 评测时间：秒级时间戳转成本地可读文本
const timeText = computed(() => {
  if (!data.timestamp) return '-'                     // 没有时间戳就显示占位符
  const d = new Date(data.timestamp * 1000)           // 注意 JS 用毫秒
  const p = (n) => String(n).padStart(2, '0')         // 补零到两位
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
         `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
})

// 分数格式化：空值显示占位符，有值保留 4 位小数
function fmt(v) {
  if (v === null || v === undefined || v === '') return '-'   // 空值兜底
  return Number(v).toFixed(4)                                 // 统一保留 4 位小数
}

// 拉取评测结果：接口无数据时返回空汇总与空明细，据此展示空表
async function load() {
  const resp = await evalResult()                     // 调接口
  Object.assign(data, resp || {})                     // 覆盖时间戳、模型名、汇总与明细
}

onMounted(() => {
  // 失败时保持空表，不阻塞页面；错误已由拦截器统一提示
  load().catch(() => {})
})
</script>

<style scoped>
/* 页面容器 */
.page {
  max-width: 1200px;       /* 表格较宽，放宽一些 */
  margin: 0 auto;          /* 水平居中 */
}
/* 标题栏右侧按钮 */
.right {
  float: right;            /* 浮动到右侧 */
}
/* 四个指标横向平铺 */
.metrics {
  display: flex;           /* 弹性布局 */
  gap: 12px;               /* 方块间距 */
  margin-top: 16px;        /* 与上方描述列表留白 */
}
/* 单个指标方块 */
.metric {
  flex: 1;                 /* 四等分 */
  padding: 12px;           /* 内边距 */
  text-align: center;      /* 居中显示 */
  background: #f5f7fa;     /* 浅灰底 */
  border-radius: 4px;      /* 圆角 */
}
/* 指标名称 */
.name {
  font-size: 13px;         /* 小字 */
  color: #606266;          /* 次要颜色 */
}
/* 指标数值 */
.value {
  margin-top: 6px;         /* 与名称留白 */
  font-size: 22px;         /* 放大突出 */
  font-weight: bold;       /* 加粗 */
  color: #409eff;          /* 主题蓝 */
}
/* 卡片之间的间距 */
.gap {
  margin-top: 16px;        /* 上间距 */
}
</style>
