# 工单 14 · RAGFlow 低质量工业 PDF 解析与信息丢失修复

> **工单编号**：人工智能 NLP-RAG-修复低质量工业 PDF 的解析与信息丢失工单 V1.1
> **工单创建**：2026-01-23 王洪荣 ｜ **工时预估**：2 人日
> **版本锚点**：RAGFlow `v0.23.1`（`pyproject.toml` = `0.23.1`，镜像 tag = `infiniflow/ragflow:v0.23.1`，两者一致）

---

## 一、项目简介

公司需搭建一套 RAG 系统，用于**中文工业技术文档的多模态推理问答**（IMDR / ITMQ 数据集，10,096 问）。
其文档以**不可编辑的图片型 PDF**（无文本层、低分辨率、复杂版面）形式提供，问题大量为**交错的文本—图像类型**。

RAGFlow 默认解析流水线在这类 PDF 上会丢失图文信息、错配图文关联。本工单定位缺陷并修复，
使规定的 6 个测试问题达到 100% 准确率。

**目标文档**：`CN100342976C.pdf`（静电除尘器专利，8 页，A4，纯图片型 PDF 无文本层，
每页 1 张整页位图约 300 DPI，MD5 `51ac11ccb34c3fd1bd95aa690f585fcc`）

**测试题**：6 道单选题，`group 1`＝纯文本（Q1/Q2）、`group 2`＝纯图像（Q3/Q4）、`group 3`＝图文混合（Q5/Q6）

| 题号 | 类型 | 标准答案 | 依据位置 |
|------|------|---------|---------|
| Q1 | 纯文本 | A | 第 1 页扉页 `[72] 发明人` |
| Q2 | 纯文本 | C | 第 1 页摘要 `[57]`（锥形部分 80–95%） |
| Q3 | 纯图像 | A | 第 7 页图 1（部件空间方位） |
| Q4 | 纯图像 | A | 第 7 页图 1（X1/X2/X3 标注）+ 第 6 页正文 |
| Q5 | 图文混合 | C | 第 7 页图 1（气流顺序） |
| Q6 | 图文混合 | B | 第 7 页图 1（D/h1/h2）+ 第 6 页正文 |

> 4/6 题依赖第 7 页工程图，纯文本切片方案在架构上无法覆盖。

---

## 二、核心结论

| 维度 | 结论 |
|------|------|
| **PDF 解析链路** | ✅ 改造成功且已固化。第 7 页图纸的空间关系被正确抽入索引，噪声下降、符号保真度提升；**源码树与运行环境同源（md5 一致）** |
| **Q3（纯图像，部件方位）** | ✅ 端到端答对 |
| **Q5（图文混合，气流顺序）** | ⚠️ 解析层正确、证据链完整，但最终选项绑定错误 |
| **端到端准确率** | **4/6 与 5/6（两轮）**，未达工单 100% 验收线 |
| **遗留差距性质** | 已不在解析链路内，位于上层问答的选项绑定 / 题面歧义 |

---

## 三、核心改动（3 处）

本工单的全部代码级改动集中在这 3 个文件，均已合入 `ragflow-main/` 源码树：

| # | 文件 | 改动内容 |
|---|------|---------|
| 1 | `ragflow-main/deepdoc/parser/figure_parser.py` | 新增 `dedup_component_triples()` 三元组去歧义后处理器；对旧镜像缺失的 `vision_llm_figure_describe_prompt_with_context` 做 ImportError 兼容降级（178 行 → 364 行） |
| 2 | `ragflow-main/rag/prompts/vision_llm_figure_describe_prompt.md` | 提示词重写：从「图表趋势分析」改为**强制输出部件方位关系与气流流向三元组**，并加入排他规则（2749 B → 4799 B） |
| 3 | `ragflow-main/docker/docker-compose.yml` | 两个 ragflow 服务的 volumes 中**新增挂载上述两个文件**，使「源码 = 运行环境」同源，不再保留独立补丁副本 |

对应的 unified diff 见 `experiments/diff/`，改动前的 v0.23.1 原版见 `experiments/baseline/`。

> ⚠️ **必须理解的环境事实**：容器内跑的是镜像烘焙的代码。只改本地源码树**不生效**（本工单实测：
> 容器内提示词 1552 B / `figure_parser.py` 178 行，与源码树完全不同）。因此第 3 项挂载是把改动送进运行时的**必要条件**。

---

## 四、目录结构

```
.
├── README.md                    # 本文件
├── .gitignore
├── docs/                        # 需求侧材料
│   ├── 工单14-需求与资源分析.md              # 需求与资源梳理（含 6 题明细、源码定位）
│   └── 人工智能NLP-RAG项目-14-...-V1.1-20260123.pdf   # 工单原件
├── deliverables/                # 正式交付物
│   ├── D1-工单任务书.md                      # 任务范围与执行约定
│   ├── D1-技术文档素材.md                    # 技术文档底稿（1622 行，覆盖工单 3 个技术问题）
│   └── D2-多轮测试结果与原因分析.md          # 多轮测试 + 原因分析 + 优化效果
├── data/source_docs/
│   └── CN100342976C.pdf         # 目标专利文档（解析对象）
├── experiments/                 # 实验过程与溯源证据（详见 experiments/README.md）
│   ├── baseline/                #   改动前 v0.23.1 原版文件
│   ├── diff/                    #   交付改动的 unified diff
│   ├── reports/                 #   A→F 各轮 + 端到端评估报告
│   ├── prompts/                 #   VLM 提示词各版本
│   ├── scripts/                 #   各轮分析 / 评估脚本
│   ├── chunks/                  #   各轮第 17/18 号切片证据
│   ├── results/                 #   端到端逐题原始作答 JSON
│   ├── groundtruth/             #   PDF 真值转写与切片对照视图
│   └── patch/                   #   中间版本产物
└── ragflow-main/                # RAGFlow v0.23.1 源码树（含上述 3 处改动）
```

---

## 五、复现步骤

### 5.1 前置条件

- Docker + Compose v2（WSL2 后端）
- 内存 ≥ 16 GB，磁盘 ≥ 50 GB
- `vm.max_map_count = 262144`（Elasticsearch 必需）

### 5.2 启动

```bash
cd ragflow-main/docker
docker compose -f docker-compose.yml up -d
```

`docker-compose.yml` 已包含源码挂载，**无需任何额外打补丁步骤**：

```yaml
volumes:
  - ../deepdoc/parser/figure_parser.py:/ragflow/deepdoc/parser/figure_parser.py
  - ../rag/prompts/vision_llm_figure_describe_prompt.md:/ragflow/rag/prompts/vision_llm_figure_describe_prompt.md
```

> 国内网络可改用 `docker-compose-CN-oc9.yml`。

### 5.3 模型配置（在 Web UI「模型提供商」中配置）

| 模型 | 用途 | 本工单实际使用 |
|------|------|--------------|
| Embedding | 向量化 | `BAAI/bge-m3@SILICONFLOW` |
| **Image-to-text（VLM）** | **图纸转文字，本工单的关键** | `qwen3-vl-plus@Tongyi-Qianwen` |
| Chat | 生成答案 | `deepseek-chat@DeepSeek`（temperature 0.1） |
| Rerank | 重排 | 未启用（避免与 3 秒响应约束冲突） |

> 注意：需求方描述的 `qwen3-v1-plus` 在实例中不存在，实际生效的是 `qwen3-vl-plus`（字母 `l`，非数字 `1`）。

### 5.4 知识库与测试

1. 新建知识库，上传 `data/source_docs/CN100342976C.pdf`
2. 解析配置：布局识别器 = `DeepDOC`，`chunk_token_num = 512`，`delimiter = "\n"`，
   `image_context_size = 0`，`table_context_size = 0`，raptor/graphrag = on
3. 建助手：`top_n = 8`，`similarity_threshold = 0.2`，`vector_weight = 0.3`
4. 逐题提问 Q1~Q6，按**选项字母**判分

---

## 六、已知限制

1. **未达 100% 验收线**：两轮端到端 4/6 与 5/6。
   - **Q5 稳定答错**：流向证据被正确抽取、被检索命中、被模型逐字引用，但最终选项绑定错误，
     认定为**题面歧义**（"先经过哪个部件"的计数起点存在两种合理解读），需在问答层或题面口径解决。
   - **Q2 存在生成波动**：两轮检索证据相同，仅生成采样不同，temperature 已设 0.1 仍出现。
2. **运行时提示词 ≠ 本地源码树**（已在本工单消除）：见第三节说明。
3. `ragflow-main/` 为上游 RAGFlow 源码（infiniflow/ragflow v0.23.1），**不含 `.git` 目录**，
   故无 commit hash，以 `v0.23.1` 作为版本锚点。
4. `ragflow-main/docker/.env` 为上游默认配置（仅含 `infini_rag_flow` 等公开默认口令，**无任何真实密钥**）。
   注意该文件被 `ragflow-main/.gitignore` 排除，若需随仓库分发请用 `git add -f`。

---

## 七、参考索引

| 主题 | 位置 |
|------|------|
| 解析任务触发链路与分块策略 | `deliverables/D1-技术文档素材.md` 第 1 章 |
| `do_handle_task` 主逻辑 | `deliverables/D1-技术文档素材.md` 第 2 章 |
| DeepDoc 解析器清单与 PDF 解析技术 | `deliverables/D1-技术文档素材.md` 第 3 章 |
| 面向本工单的缺陷定位 | `deliverables/D1-技术文档素材.md` 第 4 章 |
| 6 道测试题明细 | `docs/工单14-需求与资源分析.md` 附录 A |
| VLM 提示词与代码改动 | `experiments/diff/` |
| 逐轮实验与溯源 | `experiments/README.md` |
