# 护栏处置分级修正 —— 执行与验收报告

- 执行时间：2026-09-22 16:10–16:26
- 问题：`citation_check.py` 的逐句判定（"无引用的实质性结论"）抛通用 `CitationError`，
  而 `service.py` 对 `CitationError` 的处置是**整段替换**兜底文案 →
  一句漏标引用就丢掉整篇好回答；而更严重的"整篇零引用"反而只追加警示。
  **严重度与处置倒挂。**

## 一、改动清单（生产代码 4 个文件 + 测试 2 个文件）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `backend/app/chat/citation_check.py` | 新增异常类 `UncitedConclusionError(CitationError)`（docstring 写明"比整篇零引用轻，只警示不替换"及分级依据）；逐句判定那处 `raise CitationError` → `raise UncitedConclusionError`（**文案不变**）；模块 docstring 加 v3 变更记录；`check_citations` 的 Raises 段列明四类子类对应的处置力度。空回答、引用越界仍抛 `CitationError`（整段替换） |
| 2 | `backend/app/chat/guard.py` | 新增 `UNCITED_CONCLUSION_WARNING` + `append_uncited_conclusion_warning()`（措辞：「本次回答有部分结论未标注引用，请以引用条目为准，未标注部分请对照法源清单或官方发布文本核实」，带幂等去重） |
| 3 | `backend/app/chat/service.py` | 在 `except CitationError` **之前**插入 `except UncitedConclusionError` → 保留回答 + 追加警示 + 事件名 `citation_warning_uncited_conclusion` |
| 4 | `backend/app/chat/streaming.py` | 同上（**裁决书只点了 service.py，但 SSE 主路径有一份逐字平行的处置阶梯**；不改这里 API 路径行为不变，故一并处理） |
| 5 | `backend/tests/test_guardrail_graded.py` | 新增 `test_uncited_conclusion_keeps_answer_and_warns`（新档保留 + 事件名 + 不含 `citation_check_failed`）、`test_empty_answer_still_replaced`（空回答仍替换） |
| 6 | `backend/tests/test_chat_stream_service.py` | 新增 `test_uncited_conclusion_appends_warning_not_replace`（流式断言**不发 replace**、警示作为追加 token 下发） |

既有断言**全部保留**：引用越界 → 整段替换；零引用 → 保留 + 警示；清单外法规 → 保留 + 警示。

## 二、验收 a：端到端（`python -m app.cli.demo_ask`）

真实链路连续跑 7 题，**没有一次出现"未通过引用校验"兜底文案**，全部返回真实回答 + 末尾警示：

| 问题 | 护栏档位 | 回答 |
|---|---|---|
| 经济补偿怎么算 | `citation_warning_no_citation` | 真实回答（含分情形拆解）+ 零引用警示 |
| 试用期最长可以约定多久 | `citation_warning_no_citation` | 真实回答 + 警示 |
| 被公司辞退能拿多少赔偿金 | `citation_warning_no_citation` | 真实回答 + 警示 |
| 加班费怎么计算 | `citation_warning_no_citation` | 真实回答 + 警示 |
| 不签劳动合同有什么后果 | `citation_warning_no_citation` | 真实回答 + 警示 |
| 公司扣我工资赔偿损失每月最多扣多少 | `citation_warning_no_citation` | 真实回答 + 警示 |
| 解除劳动合同后工资什么时候结清 | `citation_warning_no_citation` | 真实回答 + 警示 |

**观察（如实报）**：当前模型在这些问题上**要么完全不标 `[n]`**（落零引用档），
所以新档（"标了一部分、漏了一句"）在自然提问里不容易命中。
端到端只能证明"不再出现兜底文案"，新档本身用下面的**真实回放实验**证明。

## 三、验收 b：全量测试

```
660 passed, 3 warnings in 15.05s
```

- 本批基线 657（`660 - 3` 条新增用例；用 `-k` 排除本批新用例后为 656/660 deselected 4，与 660 一致）
- **净增 3 条、零丢失**（函数名逐条 diff 对拍：5→7、6→7；今天 09:46–11:31 另有他处测试文件更新，
  故基线不是上一批的 652，这点已核实，避免把"别人加的测试"算进本批）

**反向证明测试真的锁住了**（`reports/b28_reverse_proof.txt`）：临时撤掉两处新分支（等于退回旧实现）
→ 2 条新用例**立刻变红**（`2 failed`），还原后文件字节一致。

## 四、验收 c：20 题评测前后对比（含稳定性对照）

三轮串行（并发会打崩上游重排），跑法：退回旧实现跑 before → 还原跑 after-1 → 再跑 after-2。

| 指标 | before（旧实现） | after-1（新实现） | after-2（新实现） | 判定 |
|---|---|---|---|---|
| **引用正确率** | **1.0000** | **1.0000** | **1.0000** | ✅ 未下降 |
| **越界引用数** | **0** | **0** | **0** | ✅ 仍为 0 |
| Recall@5 | 1.0000 | 0.9500 | 1.0000 | 上游抖动（见下） |
| MRR@10 | 0.7825 | 0.7808 | 0.7825 | 一致 |
| 误拒率 | 0.1000 | 0.0000 | 0.0500 | 上游抖动 |
| 回答无引用条数 | 16 | 16 | 14 | 上游抖动 |

**稳定性对照结论**：`after-1` 的 Recall@5 0.95 是**上游检索抖动**，与本改动无关——
同一份代码的 `after-2` 跑回 1.0（与 before 完全相同）；掉的那 1 条落在 `cross_law` 组
（0.75→1.0），而本批改动只碰护栏处置分支、完全不参与检索排序。

**档位分布（每轮 20 题，`reports/b28_tier_distribution.txt`）**：

| 档位 | before | after-1 | after-2 |
|---|---|---|---|
| 引用校验通过 | 2 | 3 | 5 |
| 零引用（保留+警示） | 14 | 16 | 13 |
| **个别结论无引用（新档）** | —（不存在此档，落入 failed） | **1** | **1** |
| 清单外法规（保留+警示） | 2 | 0 | 0 |
| **整段替换（citation_check_failed）** | **2** | **0** | **1** |

## 五、真实回放实验（本批最硬的证据，消掉 LLM 随机性）

做法：把 after 轮**真实生成的回答原文**固定住，配**该轮记录的法源清单**复现输入条件，
分别在旧实现 / 新实现下跑 `ChatService.chat`（编排、校验、护栏全走生产代码）。
每个阶段用独立子进程（同一进程里模块只导入一次，先跑旧实现会把代码缓存住）。

| 样本 | 真实回答 | 旧实现 | 新实现 |
|---|---|---|---|
| `direct-016`「公司能从我工资里扣钱赔偿吗？每月最多扣多少？」 | 854 字 | **整段替换**（→134 字兜底文案，好回答丢弃） | **保留 854 字** + 追加警示，事件名 `citation_warning_uncited_conclusion` |
| `cross-018`「最后一个月工资什么时候必须结清？解除后公司要办什么手续？」 | 779 字 | **整段替换**（→134 字） | **保留 779 字** + 追加警示 |

证据文件：`reports/b28_replay_real_answer.txt`（同输入、同法源，差异只能来自本次改动）。

## 六、验收 d：服务自检

```
check_services exit=0
Milvus=1627 == MySQL.document_chunks=1627
approved=11（其它=0）
```

## 七、边界与可逆

- **未碰 `docs/`**；全部临时脚本在 `%TEMP%/b28_bak/`，无脚本进 `backend/`
- 改前已按纪律备份：`%TEMP%/b28_bak/*.before_b28`（8 个文件含 md5），本批未新增生产文件
- 回放/反向证明脚本每次都会校验"生产代码还原字节一致"，三处实验均输出 `字节一致 OK`

## 八、两个过程发现（供参考）

1. **第 4 处改动点**：裁决书只点 `service.py`，实际 `streaming.py`（SSE 主路径）有一份逐字平行的
   处置阶梯。只改同步路径的话，前端走 API 时行为不变——已一并修。
2. **回放实验的坑**：同进程内 `app.chat.service` 只导入一次，先跑"旧实现"会把模块缓存住，
   第二段看起来仍失败（误判成"新实现没生效"）。改为子进程隔离后才拿到正确对照。

## 九、改动后护栏分级全貌（四档）

| 情形 | 异常类型 | 处置 | 是否可核查 |
|---|---|---|---|
| 回答为空 / 引用越界 | `CitationError` | **整段替换**为兜底文案 | ✗ 不可核查 |
| 整篇零引用 | `NoCitationError` | 保留 + 零引用警示 | 部分 |
| **个别句无引用的确定性结论** | `UncitedConclusionError`（本批新增） | **保留 + 未标注引用警示** | 主体可核查 |
| 提到清单外法规 | `UnknownLawCitationError` | 保留 + 清单外警示 | 主体可核查 |
