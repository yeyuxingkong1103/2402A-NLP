# 工单18 研发说明

工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流

入库前给知识库文档做体检：**这批文档现在这个样子，能不能直接进知识库？**
不能的话：哪些要 OCR、哪些是重复的、哪些有敏感信息、哪些根本打不开。

---

## 目录导航

```
研发/
├── document_quality_assessment/     ← Skill 本体（工单产出物 1、3）
│   ├── SKILL.md                     元数据层 + 指令层
│   ├── assessment_config.yaml       全部可调阈值（165 行，7 大类）
│   ├── requirements.txt
│   └── scripts/                     资源层，9 个文件 1409 行
│       ├── assess.py                主入口：遍历→解析→缓存→汇总→出报告
│       ├── summarize.py             把解析记录汇总成报告（纯内存计算）
│       ├── common.py                读配置、抽正文、断点恢复
│       ├── collect.py               格式分布 + 长度分布
│       ├── pdf_type.py              文字型/扫描型/混合型分流
│       ├── dedup.py                 MD5 精确 + SimHash 近似
│       ├── sensitive.py             敏感信息（带上下文）
│       ├── tags.py                  分类标签与主路由
│       └── report.py                JSON 报告 + HTML 简报 + 终端简报
│
├── patches/api/apps/document_app.py ← 质检 API 端点（从镜像 docker cp 出来改的）
├── workflows/
│   └── document_ingestion_workflow.json  ← 工作流 DSL（已用 RAGFlow 加载器验证）
└── ragflow/docker/docker-compose.yml     ← 带工单18 两条挂载
```

---

## 快速上手

```bash
# 1. 单元测试（不用语料，1 秒）
D:/Anaconda/envs/rag_gd/python.exe ../测试/test_skill.py

# 2. 试跑 50 份
D:/Anaconda/envs/rag_gd/python.exe scripts/assess.py <目录> --limit 50 -o /tmp/qc

# 3. 全量
D:/Anaconda/envs/rag_gd/python.exe scripts/assess.py <目录> -o <输出目录>
```

产出 `quality_report.json`（给程序）+ `quality_report.html`（给人）。

命令行参数、JSON 结构、API 端点、工作流接口 —— 见 `../设计/接口文档.md`。

---

## 读报告的推荐顺序

1. **`summary.by_primary_route`** —— 多少份要去 OCR，决定后面要花多少时间
2. **三张「待办」列表**（`details` 里）：
   - `pdf.uncertain` —— 判定存疑的，人工定夺
   - `dup.sim_pairs` —— 版本冲突，**每条都要人工确认**（SimHash 是近似匹配）
   - `sensitive.hits` —— 敏感信息，**每条都要人工确认**（带上下文）
3. **`errors`** —— 打不开的文件没进统计口径，别当它们不存在

---

## 三条要记住的事

**阈值是语料相关的。** `simhash_distance: 7` 是在 IMDR（专利）语料上扫出来的，
换语料必须重扫：`python ../测试/分析.py sweep <报告.json>`。
已知反例：年报集合那种「一套模板换数据」的语料，不同公司之间距离也能到 4~7，
用默认值会刷误报。

**扫描件比不了 SimHash。** 没有文字层就抽不出正文，正文太短的文档被
`simhash_min_chars` 挡在比较之外（**报告里会写明跳过了多少份**）。
这批文档的版本冲突要**先 OCR 再比**。MD5 那层不受影响，照常有效。

**改配置分两类，生效方式不同：**

| 类型 | 例子 | 怎么生效 |
|---|---|---|
| 判定类 | PDF 阈值、长度区间、`simhash_distance`、`tags.rules` | 直接重跑，**几秒出结果** |
| 抽取类 | 敏感信息正则、`simhash_sample_chars`、`simhash_shingle` | 要加 `--refresh` 重新解析 |

判定类之所以快，是因为缓存里存的是**事实**（字符数、每页字符数、图片覆盖率、
SimHash、敏感信息命中），判定是每次出报告现算的。

---

## 集成部署

补丁式的，不动 RAGFlow 原有逻辑。要点：

1. `docker-compose.yml` 里有带「工单18」注释的两条挂载（端点 + Skill 本体）
2. **容器 venv 缺 pymupdf，必须补装**，否则所有 PDF 被判 unreadable：
   `docker exec docker-ragflow-cpu-1 /ragflow/.venv/bin/pip install pymupdf`
   （容器重建后会丢，要重跑）
3. 工作流用 `POST /v1/canvas/set` 创建，`canvas_category` 必须是 `agent_canvas`

详见 `../部署/DEPLOYMENT.md` 和 `../设计/设计说明.md` 第四节。

---

## 交付物清单（对应工单「产出物」）

| 工单要求 | 这里的位置 |
|---|---|
| 1. 完整 Skill 源代码 | `document_quality_assessment/` |
| 2. 集成后的工作流与 API 代码 | `patches/` + `workflows/` |
| 3. 配置文件 | `assessment_config.yaml` |
| 4. 测试脚本与验证报告 | `../测试/` |
| 5. 系统演示视频 | 需人工录屏 |

**验收标准逐条对照**见 `../测试/验证报告.md` 第六节。
