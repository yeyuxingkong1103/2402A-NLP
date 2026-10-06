# API 接口文档（工单三）

> **工单编号**：人工智能NLP-RAG-PDF文档的表格解析及检索优化
> **版本**：v3.0.0（表格解析+表格感知检索）
> **Base URL**：`http://127.0.0.1:8003`
> **OpenAPI / Swagger**：`http://127.0.0.1:8003/docs`
> **接口实现**：[src/api_v3.py](../src/api_v3.py)
> **Pydantic 模型**：[src/schemas_v3.py](../src/schemas_v3.py)
> **兼容**：保留工单二 `/api/health` 端点

---

## 一、接口总览

| # | 方法 | 路径 | 说明 |
|---|------|------|------|
| 1 | `GET` | `/api/v3/health` | 健康检查 |
| 2 | `POST` | `/api/v3/ask` | 表格感知 RAG 问答 |
| 3 | `GET` | `/api/v3/tables/{doc_id}` | 获取文档所有表格 |
| 4 | `POST` | `/api/v3/upload` | 上传新 PDF 并入库 |
| 5 | `POST` | `/api/v3/evaluate` | 运行 14 题评估 |
| 兼容 | `GET` | `/api/health` | 工单二健康检查 |

---

## 二、GET /api/v3/health 健康检查

**请求示例：**
```bash
curl http://127.0.0.1:8003/api/v3/health
```

**响应 200：**
```json
{
  "status": "ok",
  "version": "v3",
  "table_collection": "rag_tables",
  "text_collection": "rag_chunks"
}
```

---

## 三、POST /api/v3/ask 表格感知问答

### 3.1 请求参数

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| question | string | 是 | - | 用户问题（中/英） |
| doc_id | string | 否 | null | 文档过滤：招股说明书1/招股说明书2 |
| top_k | int | 否 | 5 | 返回引用数（1-20） |
| use_table | bool | 否 | true | 是否启用表格检索 |
| use_rag | bool | 否 | true | false=纯 LLM 模式 |
| lang | string | 否 | null | 强制回答语言 zh/en |

### 3.2 三种执行模式

| 参数组合 | 模式 | 行为 |
|---------|------|------|
| use_rag=true, use_table=true | **工单三完整** | 表格+文本混合检索 |
| use_rag=true, use_table=false | **模拟工单二** | 仅文本检索 |
| use_rag=false | **纯 LLM** | 不检索直接生成 |

### 3.3 请求示例

```bash
curl -X POST http://127.0.0.1:8003/api/v3/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
    "doc_id": "招股说明书2",
    "use_table": true,
    "use_rag": true,
    "top_k": 5
  }'
```

### 3.4 响应 200

```json
{
  "mode": "rag_v3",
  "question": "武汉力源信息技术股份有限公司本次发行股数是多少...",
  "answer": "本次发行股数为 **1,670万股**，占发行后总股本的比例为 **25.04%**。",
  "references": [
    {
      "type": "table",
      "ref_id": "表1",
      "page": 24,
      "doc_id": "招股说明书2",
      "table_id": "tbl_011",
      "score": 0.9996,
      "preview": "股票种类、人民币普通股(A股)..."
    }
  ],
  "latency_ms": 16342,
  "breakdown": {
    "retrieve_ms": 14500,
    "llm_ms": 1842
  },
  "route": {
    "route": "table_only",
    "confidence": 0.9,
    "matched_keywords": ["发行股数", "比例"]
  },
  "retrieved_text_chunks": [],
  "retrieved_tables": [ { "table_id": "tbl_011", "rows": [] } ],
  "lang": "zh",
  "translated": false
}
```

### 3.5 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| mode | string | rag_v3 / pure_llm |
| answer | string | 最终答案（与提问同语言） |
| references | array | 引用（type=text/table，含 page/table_id/score） |
| latency_ms | float | 端到端耗时 |
| breakdown | object | retrieve_ms / llm_ms 分段耗时 |
| route | object | 路由结果 table_only/hybrid/text_only |
| retrieved_tables | array | 完整结构化表格（供前端 MD 渲染） |
| translated | bool | 英文问题是否经翻译检索 |

---

## 四、GET /api/v3/tables/{doc_id} 获取表格

**请求：**
```bash
curl http://127.0.0.1:8003/api/v3/tables/招股说明书2
```

**响应 200：**
```json
{
  "doc_id": "招股说明书2",
  "total": 230,
  "tables": [
    {
      "table_id": "tbl_003",
      "page": 2,
      "headers": ["发行股票类型", "人民币普通股(A股)"],
      "rows": [["发行股数", "1,670万股"], ["每股面值", "人民币1.00元"]],
      "caption": "发行股票类型、人民币普通股(A股)"
    }
  ]
}
```

**错误：** doc_id 不存在 → 404
```json
{ "detail": "文档 xxx 的表格文件不存在" }
```

---

## 五、POST /api/v3/upload 上传 PDF

**请求（multipart/form-data）：**
```bash
curl -X POST http://127.0.0.1:8003/api/v3/upload \
  -F "pdf=@/path/to/new.pdf" \
  -F "doc_name=新文档" \
  -F "company=某某公司"
```

**响应 200：**
```json
{
  "success": true,
  "doc_name": "新文档",
  "doc_id": "新文档",
  "text_chunks": 420,
  "table_chunks": 180,
  "message": "上传成功: 420 文本块, 180 表格块"
}
```

**处理流程：** 保存附件目录 → PyMuPDF 文本 + pdfplumber 表格 → bge-m3 向量化 → Milvus 双 collection 入库

**错误：** 非 PDF → 400；解析失败 → 500

---

## 六、POST /api/v3/evaluate 批量评估

**请求（可选自定义问题，不传则用默认 14 题）：**
```bash
curl -X POST http://127.0.0.1:8003/api/v3/evaluate \
  -H "Content-Type: application/json" \
  -d '{"modes": ["rag_v3", "pure_llm"]}'
```

**响应 200：**
```json
{
  "total_questions": 14,
  "results": [
    {
      "id": 1,
      "question": "武汉力源...发行股数...",
      "rag_v3": {
        "answer": "1,670万股，25.04%",
        "latency_ms": 71016,
        "route": "table_only",
        "table_chunks": 5
      },
      "pure_llm": {
        "answer": "1,670万股，25.01%",
        "latency_ms": 1063
      }
    }
  ],
  "summary": {
    "rag_v3_avg_latency_ms": 8979,
    "pure_llm_avg_latency_ms": 964,
    "rag_v3_table_hits": 3
  }
}
```

---

## 七、错误码

| 状态码 | 场景 |
|--------|------|
| 200 | 成功 |
| 400 | 上传非 PDF 文件 |
| 422 | Pydantic 校验失败（question 为空/None、top_k 越界） |
| 404 | doc_id 表格文件不存在 |
| 500 | 检索/解析/入库内部异常 |

---

## 八、多语言行为

| 请求语言 | 内部流程 | 回答语言 |
|---------|---------|---------|
| 中文 | 中文直接检索 | 中文 |
| English | LLM 译为中文检索 | English |
| lang 参数强制 | 覆盖自动检测 | 指定语言 |
