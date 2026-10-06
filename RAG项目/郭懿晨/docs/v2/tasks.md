# v2 检索重排任务清单

> 基于 `docs/v2/spec.md`、`docs/v2/plan.md` 拆分。**当前目录未初始化 git，无 Commit 步骤。**

## 任务总览

- [ ] 任务 1：配置扩展（rerank 配置项）
- [ ] 任务 2：SearchResult.dense 与 search(with_vectors)
- [ ] 任务 3：rerank.py 核心（粗排/精排/链路）
- [ ] 任务 4：routes_chat 接入 RetrievalChain
- [ ] 任务 5：eval 脚本 --variant + 延迟 + 对比
- [ ] 任务 6：真实 v2-m3 模型验证
- [ ] 任务 7：全量验证
- [ ] 任务 8：宪法交付物（需求文档 v2 / 版本迭代 / 架构图）
- [ ] 任务 9：评测 + 指标对比 + 汇报

## 依赖关系

```text
任务1（配置） ──→ 任务3（rerank 用配置）
任务2（dense） ──→ 任务3（粗排用 dense）
任务3（rerank） ──→ 任务4（chat 接入）──→ 任务7（全量验证）
任务3 ──→ 任务5（eval 走链路）──→ 任务9（评测对比）
任务3 ──→ 任务6（真实模型）──→ 任务7
任务7 ──→ 任务8（宪法交付物）──→ 任务9（汇报）
```

- 任务 1、2 可并行；任务 3 依赖 1、2；任务 4、5、6 依赖 3；任务 7 依赖 4、5、6；任务 8 依赖 7；任务 9 依赖 7、8。

---

## 任务 1：配置扩展

**目标：** 新增重排配置项，`top_k` 由 `retrieval_candidate_k` + `final_top_k` 取代。

**涉及文件：**
- 修改：`backend/app/config.py`
- 修改：`.env.example`
- 测试：`backend/tests/test_config.py`

**依赖：** 无

**验收标准：**
- 重排默认配置正确（rerank_enabled=True、candidate=30、coarse=10、final=6、scorer=dense_cosine）
- `top_k` 字段移除后相关默认值测试通过
- `.env.example` 同步替换 `TOP_K`

## 任务 2：SearchResult.dense 与 search(with_vectors)

**目标：** 检索结果可携带稠密向量，供粗排复用 BGE-M3 信号。

**涉及文件：**
- 修改：`backend/app/vector_store.py`
- 测试：`backend/tests/test_vector_store.py`

**依赖：** 无

**验收标准：**
- `SearchResult` 新增可选 `dense` 字段，默认 None
- `search(with_vectors=True)` 返回 dense（Qdrant named vector 提取）
- 内存实现与兜底路径 dense 为 None

## 任务 3：rerank.py 核心

**目标：** 实现粗排打分器、粗排器、精排器、RetrievalChain。

**涉及文件：**
- 新增：`backend/app/rerank.py`
- 测试：`backend/tests/test_rerank.py`

**依赖：** 任务 1、任务 2

**验收标准：**
- 三种 scorer（dense_cosine / rrf_score / hybrid）行为正确，dense 缺失时回退 rrf
- 粗排排序 + 截断正确
- 精排批处理、排序、top_k、分数覆盖正确（用 fake FlagReranker）
- `RetrievalChain.retrieve`：v2 全链路 + v1 退化路径 + 空候选
- 模型路径缺失时抛可诊断错误

## 任务 4：routes_chat 接入 RetrievalChain

**目标：** `/api/chat` 改经检索重排链路。

**涉及文件：**
- 修改：`backend/app/routes_chat.py`
- 测试：`backend/tests/test_routes_chat.py`

**依赖：** 任务 3

**验收标准：**
- chat 走 `build_retrieval_chain(...).retrieve(question)`
- 空结果走现有兜底
- 测试的 FakeSettings 换成新配置字段

## 任务 5：eval 脚本 --variant + 延迟 + 对比

**目标：** 评测支持 v1/v2/both，采集检索延迟并产出对比报告。

**涉及文件：**
- 修改：`scripts/run_sf6_eval.py`
- 测试：`backend/tests/test_sf6_eval_runner.py`

**依赖：** 任务 3

**验收标准：**
- `--variant v1|v2|both` 与 `SF6_EVAL_VARIANT` 环境变量生效
- 每个 item 记录 `latency_s`，summary 含 `avg_latency_s`
- 产出 `{variant}_metrics.json` 与 `v1_vs_v2_comparison.json`

## 任务 6：真实 v2-m3 模型验证

**目标：** 确认 FlagEmbedding 兼容性，真实加载 v2-m3 打分。

**涉及文件：**
- 新增：`backend/tests/test_rerank_real.py`
- 可能修改：`requirements.txt`

**依赖：** 任务 3

**验收标准：**
- `FlagReranker` 可导入（不兼容则升级锁定版本并记录）
- 真实模型打分在 [0,1] 区间
- 记录 GPU 单次批打分延迟

## 任务 7：全量验证

**目标：** 全部测试通过、API 可导入、规格覆盖核对。

**涉及文件：**
- 全部已改动代码与测试

**依赖：** 任务 4、5、6

**验收标准：**
- `pytest backend/tests -v` 全部通过（含真实模型测试）
- `from backend.app.main import app` 可导入
- FR-001~FR-009、NFR、边界条件逐条核对通过

## 任务 8：宪法交付物

**目标：** 升级需求文档至 v2、补版本迭代、新建架构图-v2。

**涉及文件：**
- 修改：`docs/需求说明.md`
- 新增：`docs/版本迭代.md`
- 新增：`docs/架构/架构图-v2.md`
- 修改：`README.md`

**依赖：** 任务 7

**验收标准：**
- 需求文档含版本头（v2）、版本迭代表、FR15~FR18；范围外移除「reranker 精排」
- `docs/版本迭代.md` 记录 v1/v2
- `docs/架构/架构图-v2.md` 六层架构含重排链路

## 任务 9：评测 + 指标对比 + 汇报

**目标：** 跑 v1/v2 对比评测，记录指标对比，向用户汇报。

**涉及文件：**
- 产出：`eval/results/v1_metrics.json`、`eval/results/v2_metrics.json`、`eval/results/v1_vs_v2_comparison.json`
- 追加：`docs/指标对比/优化指标对比.md`

**依赖：** 任务 7、任务 8

**验收标准：**
- 真实评测产出对比文件
- 指标对比文档追加 v2 记录（含 safety 集缺口备注）
- 向用户汇报改动清单与指标变化
