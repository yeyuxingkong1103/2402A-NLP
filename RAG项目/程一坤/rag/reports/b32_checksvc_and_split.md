# 批次 32：check_services 假绿灯修复 + citation_check 拆分 + 统一处置阶梯抽取

日期：2026-09-22　|　改动前备份：`%TEMP%/b32_bak/*.before_b32`（5 文件 md5 已记）

## ① check_services.py 假绿灯修复

根因（用户核实）：`report()` docstring 承诺"失败时登记到 failures"，实现没有 append；
只有显式 `failures.append` 的检查影响退出码 → 打印 ❌ 但退出码 0（LLM 探针空内容为真实案例）。

改动（`scripts/check_services.py`，md5 b8c4f674→6acb8e7c，433 行）：
1. `report()` 加 `key` 参数（默认用 name），`not ok` 时统一 `failures.append(key or name)`；
2. 收敛全部 17 处显式 `failures.append`（现仅剩 report() 内 1 处），失败键与展示名不同的
   场景（容器名、milvus_collection/consistency/stats、docker/redis/mysql/milvus、embedding/
   reranker/llm）改传 `key=`，失败键取值与改前完全一致，无重复计数；
3. LLM 探针 `max_tokens` 16→64；空 content 时 detail 写明"返回空内容（可能 max_tokens 过小或模型异常）"；
4. Embedding/Reranker/LLM 的 ok 路径 report 补 key——非异常失败（空 results/空 content）现在也登记。

调用点审计：29 处 report() 逐个核对，所有 ok=False 路径均经 report 登记，无残留直连 append。

自测 `backend/tests/test_check_services.py`（5 用例，importlib 按路径加载 + 伪造 urlopen/embedding 模块）：
report 登记 key/默认 name、main 退出码 1/0、Reranker 空 results、LLM 空 content。
反向证明：撤掉 report 内 append → 4 条失败登记用例红（含两条假绿灯路径）→ 字节还原 → 5 passed。

验收：
- (a) 正常全绿 exit 0（b32_check_normal.txt）
- (b) 假 LLM 地址 → `❌ LLM` + failures=llm + **exit 1**（b32_check_fakebase.txt）
- (a') 真实 --with-api 三探针全绿 exit 0（b32_check_withapi.txt；b31 时 LLM 0 字假红不再复现）

## ② citation_check 拆分（300 顶格 → 两文件）+ 统一处置阶梯

### 拆分
- `app/chat/citation_check.py`（300 行 → **158 行**）：结构性校验入口——空回答、[n] 提取与
  越界、零引用分支、四类异常抛出时机；旧公共名全部转发，3 个测试文件的既有 import 不变。
- `app/chat/citation_language.py`（新建，**205 行**）：表述词表、法规名后缀白名单（注释原样保留，
  含案例材料边界）、归一化、`find_unknown_laws()`、`scan_uncited_conclusion()`（逐句块原样搬移）。

行为不变证据：
- **40 合成用例对拍**（b32_paridad_check.txt）：旧实现 vs 新实现，异常类型+消息逐字一致，0 不一致；
- **真实回答回放**（b32_replay_eval_answers.txt）：b30/b29 两轮评测的 40 条真实回答逐条对拍，0 不一致；
- 既有 36 条引用校验测试全绿。

### 统一处置阶梯
- `guard.py`（247→287 行）新增 `resolve_citation_outcome(answer, error) -> (answer, event_name)`，
  事件名前缀收拢为 4 个模块常量（取值与拆前逐字一致，`前缀: {error}` 格式不变）；
- `service.py`（235→212 行）/`streaming.py`（211→193 行）：删除两份平行 except 阶梯，各留
  一个 `except CitationError` 委托共用函数；streaming 只决定"怎么送达"（警示类补发增量 /
  硬违规 replace+finish），交付语义与拆前逐行等价。
- 新测试 `test_citation_outcome_shared.py`（12 用例）：四类异常的结果与事件名逐字断言、
  幂等（重复处置不重复追加）、同异常结果一致、**AST 结构锁**——service/streaming 各恰一次
  resolve 调用且不得内联任何事件名字符串。
- 反向证明（b32_reverse_proof_outcome.txt）：改坏 guard 事件名取值 → 事件名用例红；
  往 service.py 塞回内联事件名 → 结构锁用例红；字节还原后 12 passed。

## 验收汇总

| 项 | 要求 | 实测 |
|---|---|---|
| 全量测试 | 不减 | **688 passed**（671 + 5 check_services + 12 outcome_shared） |
| check_services | 全绿 exit 0 / 假探针 exit≠0 | ✅ 0 / ✅ 1 |
| 20 题评测 | 与拆前逐条一致 | 检索指标与 b29 逐位相同（Recall@5 1.0 / MRR 0.7825）；40 条真实回答校验对拍 0 不一致；本轮引用正确率 1.0、越界 0、无引用 1/20（= direct-012 LLM 幻觉编 [6] 被正确整段替换，属预期处置非回归） |
| demo_ask | 正常 | `citation_check_passed`，[1]~[5] 与法源一一对应（b32_demo_ask.txt） |
| SSE | 正常 | 10/10（b32_sse.txt，探针数据已清） |

改动文件 md5（实测）：check_services 6acb8e7c / citation_check 6408ecbf / citation_language e8ca8212 / guard 7369a281 / service bc073579 / streaming 20c01035 / test_check_services 17cbce65 / test_citation_outcome_shared 9fcf20a8。全部 CRLF=0（实测核对）；backend/、evaluation/ 无临时脚本；未动 docs/。

## 备注
- Edit 工具本次把 service.py/streaming.py 翻成 CRLF（212/193 处），已按字节修回 0 并重核 diff 只含本批编辑——下批改源码优先用脚本 read_bytes/write_bytes。
- `direct-012` 本轮被整段替换是 LLM 幻觉越界引用的正确处置，若未来多轮频繁出现可考虑提示词侧再收紧，本轮不处理。
