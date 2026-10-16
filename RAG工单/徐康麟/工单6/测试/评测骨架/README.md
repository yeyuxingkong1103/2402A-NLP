# 测试/评测骨架 —— T6 评测与留痕骨架（**结构契约；实跑入口见 `测试/在线/run_16q_eval.py`**）

> 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
> 归属：测试工程师（T6 执行）；**不碰 `研发/scripts/**` 与 `研发/app/**`（研发侧地盘）**。
> 评测驱动：本目录 `evaluate_16q_skeleton.py`（骨架）；**正式版在 `测试/在线/run_16q_eval.py`**
> （参数：`--base-url / --out-dir / --run-tag / --top-k / --timeout / --use-cache / --no-warmup / --suites`）。
> 状态：**骨架**。本目录所有文件里凡与「成绩/延迟/命中率」相关的字段一律为 `None`
> 或文字「待 T6 实跑填充」，**不得预填任何占位数、示例数字或结论性语句**。

## 0. 硬约束（captain 下达，T6 必须遵守）

1. **不预填数字**：输出结构中涉数字段一律 `None` + 「待 T6 实跑填充」；
2. **不写结论**：不得出现「准确率 ≥90% 达标」这类判断；达标与否只能由实跑数据 + `设计/验收标准.md` 阈值推出；
3. **数字只有一个来源**：真实调用引擎（HTTP/CLI）后的**实测值**；任何缓存/重放数据必须标注口径；
4. **图像题不得声称 OCR**：本机无 OCR（见 `golden.chart.no_ocr_note`）；
5. **两个 K 分列**：生成阶段 Top-K=3（T2/T5 配置）与命中判定 K=`record["min_hit_top_k"]`=5
   必须在报告中分别标注来源与取值。

## 1. 本目录文件

| 文件 | 用途 | 状态 |
| --- | --- | --- |
| `qa_results_schema.json` | 逐题结果与汇总的**输出结构契约**（数字全为 null） | 骨架 |
| `evaluate_16q_skeleton.py` | 评测 CLI：参数解析 + golden 加载 + 逐题调用适配器 + 结果落盘 | 骨架（未接引擎） |
| `screenshot_16q_skeleton.py` | 截图留痕清单与命名规范（**不驱动浏览器**，T6 用它生成清单） | 骨架 |
| `测试报告模板.md` | 测试报告表格骨架（结论列全为「待 T6 实跑填充」） | 骨架 |
| `conftest_16q_template.py` | T6 在线测试的 pytest 夹具模板（判分 / 检索命中 / golden 索引） | 骨架 |

## 2. 数据来源（唯一真值，禁止另立）

```python
import importlib.util
spec = importlib.util.spec_from_file_location("golden", r"测试/common/golden.py")
golden = importlib.util.module_from_spec(spec); spec.loader.exec_module(golden)

recs = golden.load_golden_index()          # 16 题，key = 原始 id（5,6,1,2,...207）
judge = golden.judge_answer(rec, answer)   # 回答判对（确定性，非 LLM）
hit = golden.check_retrieval_hit(rec, chunks)   # 检索命中（前 K 条内命中即算，K=rec["min_hit_top_k"]）
```

- golden 主真值：`研发/data/eval/golden_qa.jsonl`（镜像 `测试/测试数据/golden_qa_16.jsonl`）
- 判分口径全文：`测试/测试数据/golden_复核报告.md` §6（**测试报告必须照抄其判分说明**）
- 逐题完整性要求：`record["answer_form"]`；名称变体：`record["answer_keywords_any_of"]`

## 3. 输出契约（`--out-dir` 目录；与正式版 `测试/在线/run_16q_eval.py` 参数名一致）

| 文件 | 内容 |
| --- | --- |
| `qa_results.csv` | 逐题：题号/id/题型/问题/系统答案/golden 答案/是否判对/证据页/引用页/检索命中/clip 页/`latency_total_ms`/`latency_first_token_ms`/口径标记 |
| `qa_results.jsonl` | 同上，逐题一行，附 `judge_detail`（逐关键字命中明细）与 `retrieval_detail` |
| `accuracy_report.json` | 汇总：准确率、判对题数、检索命中率、延迟 **P50/P95/最大**、口径与统计量说明 |
| `run_meta.json` | 运行元信息：时间、run-tag、服务地址、模型、K 的两个取值、是否预热、是否独占 |

**未实跑时**：`evaluate_16q_skeleton.py --transport stub` 会把上述文件写出，
但所有数字字段为 `null`、`note` 为「待 T6 实跑填充」——用于先冻结报告结构，**不产生任何成绩**。

## 4. 复现命令（T6 实跑时）

```powershell
cd E:\gao6gongdan\工单4

# 骨架自检（不接引擎，只产出 null 结构与截图清单；产物默认写系统临时目录）
pwsh -NoProfile -File run_py.ps1 测试/评测骨架/evaluate_16q_skeleton.py --transport stub
pwsh -NoProfile -File run_py.ps1 测试/评测骨架/screenshot_16q_skeleton.py --list

# T6 正式评测（引擎就绪后，由 T6 落地具体端点与预热/独占自检）
pwsh -NoProfile -File run_py.ps1 测试/评测骨架/evaluate_16q_skeleton.py --transport http --base-url http://127.0.0.1:8123 --run-tag accept
```

> ⚠️ **交付纪律（勿破）**：stub 产物默认写**系统临时目录**，**不得**留在 `测试/**`——
> 一份「成绩字段为 null」的 `qa_results.csv` 若出现在交付目录，出报告时极易被误当成实测结果。
> 若确需落盘，请显式传 `'--out-dir=<临时路径>'`（注意用引号，否则 PowerShell 会把 `--out-dir` 当成自己的参数），
> 并在交付前删除。`--transport http` 的实测产物默认写 `测试/测试数据/评测_<run-tag>/`。

## 5. 待办（T6 派工后逐项落地）

- [ ] 接入真实端点与请求/响应字段映射（按 `设计/接口设计.md`）
- [ ] 预热与独占自检证据（`ollama ps` 无其它模型加载、到 11434 的 ESTABLISHED = 0）
- [ ] 负例 4 条（拒答红线 N1~N4）与中英文各 ≥1 题
- [ ] 逐题截图（`测试/测试截图/`，命名规范见 `screenshot_16q_skeleton.py`）
- [ ] 把 §6 判分口径原文照抄进 `测试/测试报告.md`

## 6. 报告口径硬要求（captain 指派，T6 逐条落地）

1. **检索命中率按通道分列**：BM25 / 向量 / 任一通道**三列并报**，**禁止只报合并值**——
   合并会掩盖「向量召回仅 11/16」这一真实短板（上游自检值：BM25 页级 16/16、向量页级 11/16；
   关键词命中 BM25 15/16、向量 13/16）。分列后才能判断「答错」是**检索未命中**还是**生成未写对**。
2. **命中判定分三层（captain 裁定，不可合并）**：`golden.check_retrieval_hit_layers()` 返回
   ① `hit_complete`（权威层，**唯一命中率主指标**：chunk 含完整证据原文）／
   ② `hit_any_evidence`（诊断层A：chunk 含 complete ＋ supporting 的任一证据原文）／
   ③ `hit_page_level`（诊断层B：chunk.page ∈ evidence_pages，**禁止**用作命中率）。
   层②≠层③：前者要求证据原文真的落在 chunk 里。实例（id 5，complete=[39]、supporting=[40,113]）：页 40 chunk
   → ① False ｜ ② True ｜ ③ True；页 113 的**无关内容** chunk → ① False ｜ ② False ｜ ③ True。
   自检：`pwsh -NoProfile -File run_py.ps1 测试/评测骨架/三层命中口径自检.py`。
3. **索引统计用刷新后的现役值**：总量 **2676** = text 1981 / table 618 / image **77**；图像块 77 = **PDF1 38 + PDF2 39**
   （picture 43 / diagram 12 / chart 22）；vectors (2676,1024)。旧值 2701/102 系首轮表格误判框图所致，**已作废**。
   引用前须复核 `研发/data/index/index_manifest.json`，并在报告中记录 `chunk_total` 以追溯版本；
   **取数须在 T5 重载索引之后**（索引跨文件非事务切换，存在数百毫秒版本混读窗口）。
4. **证据权重口径沿用 T4 定义**：diagram / chart = 1.0（强证据）；**picture = 0.4（仅 CLIP 零样本标签，弱证据）**，
   不得与正文同权。
5. **复核上游修复项**：`vector_top10[].page` 曾误标名次，现应同时有 `rank`（1-based）与 `page`（真实物理页，
   与 `chunk_id` 的 `pN` 一致）。T3 已只读抽查确认闭环（`rank=1, page=39, chunk_id=pdf2-p39-img-x0`），T6 仍需全量抽查。
6. **图像题不得声称 OCR**；上述 5 条 + 截图命名规范逐条留痕。
