# results 目录说明（结果文件格式）

**工单编号：人工智能NLP-RAG-图像内容解析及检索优化**

本目录是工单04 的**全部运行产物**，由 `src/` 下的脚本生成，可随脚本重跑覆盖。
所有 JSON 均为 UTF-8、`ensure_ascii=False`，可直接用编辑器查看中文。

> 目录中的 `images/`、`cache/` 与各结果文件在首次运行脚本前不存在，属正常现象。

---

## 一、文件清单

| 文件 | 生成脚本 | 对应工单要求 |
|------|----------|--------------|
| `images/<文档名>/*.png` | `image_extractor.py` | 抽出的图像（含矢量图整页渲染） |
| `image_inventory.json` / `.md` | `image_extractor.py` | 图片清单：页码/尺寸/来源/CLIP 分类 |
| `image_descriptions.json` / `.md` | `image_semantic_parse.py` | 图像语义描述 + 结构化数据 |
| `index_stats.json` | `build_multimodal_index.py` | 多模态知识库索引统计 |
| `image_qa.json` / `.md` | `image_qa.py` | id5/id6：命中图像 + 描述 + 答案 + 精确度 |
| `image_ablation.json` / `.md` | `compare_with_without_images.py` | **消融对比实验（核心证据）** |
| `evaluation_multimodal.json` / `.md` | `run_evaluation.py` | 16 问完整评估（含图像） |
| `evaluation_text.json` / `.md` | `run_evaluation.py` | 16 问评估（不含图像，对照） |
| `cache/semantic_cache.json` | `image_semantic_parse.py` | 多模态解析缓存（断点续跑用） |

---

## 二、JSON 字段说明

### 1. `image_inventory.json`

```jsonc
{
  "workorder": "人工智能NLP-RAG-图像内容解析及检索优化",
  "generated_at": "2026-02-06 10:20:31",
  "min_side": 200,            // 有效图像最小边长（像素）
  "render_dpi": 200,          // 矢量图整页渲染分辨率
  "n_images": 12,
  "n_embedded": 10,           // 内嵌位图数量
  "n_page_render": 2,         // 矢量图整页渲染回退数量
  "doc_stats": { "招股说明书2": { "pages": 350, "n_embedded": 10,
                                 "n_page_render": 2, "n_figure_page": 4 } },
  "images": [
    {
      "image_id": "招股说明书2#p39#fullpage",
      "doc": "招股说明书2",
      "page": 39,
      "source": "embedded | page_render",   // 来源：内嵌位图 / 整页渲染回退
      "file": "results/images/招股说明书2/p39_fullpage.png",
      "abs_file": "C:\\Users\\...\\p39_fullpage.png",
      "width": 1654, "height": 2339,
      "xref": null,                          // page_render 无 xref
      "bytes": 412345,
      "caption": "1、公司组织结构图",         // 该页抽取到的题注
      "figure_page": true,
      "figure_reason": "绘图对象密集|无可用内嵌图",
      "dup_xref": false,
      "clip_labels": [ {"label": "组织结构图", "score": 0.93} ]  // CLIP 分类结果
    }
  ]
}
```

### 2. `image_descriptions.json`

```jsonc
{
  "extractor": "CLIP(零样本分类) + 多模态大模型(语义描述/结构化抽取)",
  "vlm_backend": "OpenAI 兼容多模态接口（qwen-vl-max）",
  "n_images": 6, "n_ok": 6, "n_page_render": 2,
  "images": [
    {
      "image_id": "招股说明书2#p39#fullpage",
      "doc": "招股说明书2", "page": 39, "source": "page_render",
      "file": "results/images/招股说明书2/p39_fullpage.png",
      "caption": "1、公司组织结构图",
      "kind": "org",                 // org=组织结构图专用 prompt；auto=通用图表
      "chart_type": "组织结构图",
      "clip_labels": [...],
      "vlm_description": "图标题：公司组织结构图\n\n层级结构：\n- 股东大会\n ...",
      "chart_data": {},              // kind=auto 时为结构化 JSON
      "hierarchy_facts": [           // 由层级树程序化展开（计数确定）
        "销售部 的直接下级（4 个）：电话及网络销售部、渠道销售部、大客户销售部、国际贸易部",
        "大客户销售部 的直接下级（6 个）：珠海销售处、深圳销售处、北京销售处、武汉销售处、广州销售处、成都销售处"
      ],
      "error": "",
      "cached": false,
      "parse_seconds": 6.4,
      "retrieval_text": "[图像] 《招股说明书2》第39页｜类型：组织结构图｜...（入库文本）"
    }
  ]
}
```

> `retrieval_text` 是**真正写入向量库与 BM25 的文本**，由「头信息 + 语义描述 + 结构化数据 + 层级事实」拼成。

### 3. `index_stats.json`

```jsonc
{
  "with_images": {
    "collection": "wo04_multimodal",
    "docs": [ {"name": "招股说明书2", "pages": 350, "blocks": 812,
               "chunks": 903, "image_chunks": 6, "seconds": 61.2} ],
    "n_chunks": 1650,
    "chunk_types": { "text": 1520, "table": 124, "image": 6 },
    "n_vectors": 1650, "n_bm25_docs": 1650,
    "bm25_path": "...\\data\\index\\bm25_wo04_multimodal.pkl",
    "image_blocks_filled": 6, "image_blocks_dropped": 0,
    "build_seconds": 128.4, "built_at": "2026-02-06 10:25:00"
  },
  "text_only": { "...": "结构同上，无 image 块" }
}
```

### 4. `image_qa.json`（演示与验收主材料）

```jsonc
{
  "n_questions": 2, "n_correct": 2, "accuracy": 1.0,
  "results": [
    {
      "id": 5,
      "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？",
      "ground_truth": "销售部下设 4 个部门……大客户销售部下设 6 个销售处……",
      "hit_images": [                     // ★ 命中的图像
        { "doc": "招股说明书2", "page": 39, "score": 0.87,
          "chunk_id": "招股说明书2-ima-00003",
          "image_files": ["results/images/招股说明书2/p39_fullpage.png"],
          "image_source": ["page_render"],
          "semantic_description": ["图标题：公司组织结构图……"],  // ★ 语义描述
          "chart_data": [{}],
          "hierarchy_facts": ["销售部 的直接下级（4 个）：……"],
          "text_snippet": "[图像] 《招股说明书2》第39页｜类型：组织结构图｜……" }
      ],
      "answer": "销售部由 4 个部门构成……",                      // ★ 生成的答案
      "citations": [ {"page": 39, "type": "image", "chunk_id": "..."} ],
      "precision": {                       // ★ 检索精确度
        "context_precision": 0.9167,       // RAGAS 口径：有用片段是否排在前面
        "要点命中率": 1.0,                  // 命中要点数 / 要点总数
        "命中要点": ["渠道销售部", "..."],
        "漏答要点": [],
        "要点判定": true,                   // 全部要点命中才算答对
        "图像页命中": true,                 // 是否召回了期望图页（39 / 72）
        "期望图页": [39]
      },
      "retrieved_pages": [39, 40, 38, 112, 71],
      "retrieved_chunks": [                 // Top-k 检索片段（含分数与来源）
        { "rank": 1, "chunk_id": "...", "doc": "招股说明书2", "page": 39,
          "type": "image", "score": 0.031, "snippet": "[图像] ……" }
      ],
      "timings": { "retrieve_s": 0.21, "total_s": 4.83,
                   "detail": { "vector_recall": 0.11, "fulltext_recall": 0.03,
                               "fusion": 0.001, "rerank": 1.62, "total": 0.21 } }
    }
  ]
}
```

### 5. `image_ablation.json`（核心证据）

```jsonc
{
  "summary": {
    "n_questions": 16,
    "A_不含图像": { "collection": "wo04_text", "accuracy": 0.75,
                    "correct": 12, "n_judged": 16,
                    "retrieval_latency_avg": 0.19, "total_latency_avg": 4.2,
                    "n_chunks": 1600 },
    "B_含图像":   { "collection": "wo04_multimodal", "accuracy": 0.875,
                    "correct": 14, "n_judged": 16,
                    "retrieval_latency_avg": 0.21, "total_latency_avg": 4.4,
                    "n_chunks": 1606, "n_image_chunks": 6 },
    "accuracy_delta": 0.125,
    "retrieval_latency_delta": 0.02,
    "image_questions": { "id5": { "不含图像": false, "含图像": true },
                         "id6": { "不含图像": false, "含图像": true } }
  },
  "questions": [
    { "id": 5, "question": "...", "不含图像_正确": false, "含图像_正确": true,
      "变化": "提升", "不含图像_漏答要点": ["6 个销售处", "珠海", "..."],
      "含图像_漏答要点": [],
      "不含图像_答案": "根据提供的文档内容，未能找到该问题的答案。",
      "含图像_答案": "销售部由 4 个部门构成……",
      "不含图像_检索页码": [40, 41], "含图像_检索页码": [39, 40],
      "含图像_命中图像块": [ {"page": 39, "type": "image", "snippet": "..."} ] }
  ]
}
```

### 6. `evaluation_*.json`

```jsonc
{
  "summary": {
    "collection": "wo04_multimodal",
    "n_questions": 16, "n_judged": 16, "accuracy": 0.94, "correct": 15,
    "retrieval_latency_avg": 0.21, "retrieval_latency_max": 1.8,
    "total_latency_avg": 4.4, "total_latency_max": 9.1,
    "n_under_3s_total": 6, "n_under_3s_retrieval": 16,
    "image_questions_correct": { "5": true, "6": true },
    "ragas": { "faithfulness": 0.93, "answer_relevancy": 0.95,
               "context_precision": 0.88, "context_recall": 0.91,
               "answer_correctness": 0.90 }      // 仅在 --ragas 时出现
  },
  "details": [
    { "id": 5, "question": "...", "answer": "...", "是否答对": true,
      "命中要点": ["..."], "漏答要点": [], "命中图像块": [ {"page": 39} ],
      "检索页码": [39, 40], "检索耗时(s)": 0.21, "总耗时(s)": 4.83,
      "所用检索耗时明细": { "vector_recall": 0.11, "rerank": 1.62 } }
  ],
  "records": [ /* rag_core.evaluate.EvalRecord.to_dict() 的标准字段 */ ]
}
```

---

## 三、使用建议

* **验收看三个文件**：`image_qa.md`（图像问答效果）、`image_ablation.md`（前后对比）、
  `evaluation_multimodal.md`（16 问指标）；
* **复现**：按 `docs/用户手册.md` 的顺序重跑 7 个脚本即可重新生成全部文件；
* **人工核验**：`images/招股说明书2/p39_fullpage.png` 与 `p72_*.png` 可直接打开与图中数字对照；
* **口径说明**：`accuracy` 为"答案要点全部命中"的严格口径；`retrieval_latency` 不含 LLM 生成，
  `total_latency` 含生成，两者的区别见 `docs/技术文档.md` §7.1。
