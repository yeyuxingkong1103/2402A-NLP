# 批次 31：落库写入真实模型名 + 案例类标题边界注释

日期：2026-09-22。范围：裁决 1（persist 写真实模型名）、裁决 2（citation_check 案例类注释）。
裁决 3（`guard.resolve_citation_outcome` 抽取）按约定**本批未做**，留待下一批，且前置是先把
`citation_check.py` 按职责拆到两文件各 ≤250 行。

---

## 0. 结论速览

| 项 | 结果 |
| --- | --- |
| ① 落库 model 取当轮配置 | ✅ 真实 SSE 落库 `model='deepseek-flash'`（与当轮配置逐字相等） |
| ① 历史 `model=NULL` 不回填 | ✅ 跑前 8 行 → 跑后 8 行；全表非 NULL 仅从 0 → 1（就是本轮那一行） |
| ② 案例类边界注释 | ✅ 仅新增 1 行注释，**AST 与改前完全一致**（行为零改动） |
| 全量测试 | ✅ **671 passed**（基线 665 + 新增 6） |
| check_services | ✅ exit 0（`--with-api` 与默认模式各跑一次，均 exit 0，见 §6 附带说明） |
| 改动范围 | ✅ 3 个生产文件 + 1 个新测试文件；临时脚本全在 `%TEMP%/b31_bak/`；未动 `docs/` |

---

## 1. ① persist 写真实模型名

### 1.1 改动（改前均备份至 `%TEMP%/b31_bak/*.before_b31`，CRLF=0）

`backend/app/api/chat_persistence.py`（243 行）

```diff
-            # LLM 客户端未暴露模型名，禁止伪造，落库为 NULL
-            model=None,
+            # 模型名取当轮生效配置（批次 31）：记真实值，模型变更才可追溯（此前写死 NULL，
+            # 导致历史消息无法回溯"这条回答是哪个模型生成的"）
+            model=current_llm_model(),
```

新增读取函数（与同文件 `_session_summary_enabled` 的同款"懒读配置 + 降级"写法）：

```python
def current_llm_model() -> str | None:
    """读当轮生效的 LLM 模型名（懒读配置，测试可通过环境变量覆盖）。"""
    try:
        from app.core.config import settings
        return settings.llm_model or None   # 空串按"未配置"处理
    except Exception:  # noqa: BLE001
        return None                          # 配置不可读不阻断回答，禁止伪造
```

**为什么由持久化层读配置，而不是让 LLM 客户端回传**：模型名是"当轮配置"的事实，落库只需要它；
为它把字段一路穿过检索/生成链路，等于把配置耦合进能力层，收益不抵成本。

> 附带一处**注释准确性修正**（`backend/app/chat/chat_store.py`，279 行，单行内容替换、行数不变）：
> `- model: 生成模型名（LLM 客户端未暴露时为 None，禁止伪造）` →
> `- model: 生成模型名（由调用方从当轮配置取，取不到时为 None，禁止伪造）`。
> 该文件**去 docstring 后 AST 与改前完全一致**，属纯注释修正。不认可可撤。

**未改动**：write 路径的 user 消息 `model` 仍为 NULL（提问不由模型生成）；历史行一律不动。

### 1.2 新增测试 `backend/tests/test_chat_persistence.py`（145 行，6 个用例）

| 用例 | 锁住什么 |
| --- | --- |
| `test_persist_turn_writes_model_from_current_config` | 落库入参 `model` == 当轮配置值，且非 None |
| `test_persist_turn_model_follows_config_change` | 换配置落库值就跟着换（证明"取自配置"而非写死字符串） |
| `test_persist_turn_keeps_other_fields_untouched` | 只改 model：question/answer/sources/message_id 逐字不变 |
| `test_current_llm_model_returns_none_when_unconfigured` | 空串 / 属性缺失一律返回 None，不抛错 |
| `test_persist_turn_still_persists_when_model_unconfigured` | 取不到模型名时照常落库（回答流不受影响） |
| `test_current_llm_model_reads_env_backed_settings` | 走真实配置装载链路（屏蔽 .env + 注入环境变量）能取到值 |

### 1.3 反向证明（`b31_reverse_proof.txt`）

只把 `model=current_llm_model(),` 退回 `model=None,`（**保留新函数定义**，避免 import 期报错掩盖行为）：

```
[1] 新实现跑新用例            : 6 passed  exit=0
[2] 退回旧行为（仅 model 一行）: 2 failed  exit=1
      FAILED test_persist_turn_writes_model_from_current_config
      FAILED test_persist_turn_model_follows_config_change
      E  assert [None, None] == ['model-a', 'model-b']
[3] 还原后 md5 = 5fed85eb6fef85df7648695e5130f0dc（与改后一致）
    CRLF 数 = 0
[4] 还原后复跑新用例          : 6 passed  exit=0
反向证明成立
```

### 1.4 真实 SSE 验收（`b31_sse_model_accept.txt`，10/10 通过）

做法：真起后端（uvicorn@127.0.0.1:**8124**，8123 被占用、8000 被 Docker Desktop 占用）→ 真 Redis
签发探针令牌（`chat_sessions.user_id` 只存 user_key、无外键，因此不必造 users 行）→ 真打
`POST /api/v1/chat/stream` 拉完整 SSE → 真 MySQL 查落库行。子进程 stdout/stderr 重定向到日志文件，
跑前剔除 `http_proxy/https_proxy`。

**落库原始行（本次唯一要求的贴图）**：

```
[落库] SELECT id, role, model, CHAR_LENGTH(content), created_at
       FROM chat_messages WHERE session_id=24 ORDER BY id DESC;
   id=44  role=assistant  model='deepseek-flash'  len=760  created_at=2026-09-22 09:17:04
   id=43  role=user       model=None              len=7    created_at=2026-09-22 09:17:04
```

其余核对项：

```
SSE：status=200  事件={message_start:1, token:458, citation:5, message_end:1}  回答长度=760
model 列定义：model varchar(128)        ← 落库值不会被截断
跑前：历史 model IS NULL（排除探针会话）= 8；全表非 NULL = 0
跑后：历史 model IS NULL（排除探针会话）= 8；全表非 NULL = 1
✅ assistant model == 当轮配置 'deepseek-flash'
✅ assistant model 非 NULL     ✅ user model 仍为 NULL     ✅ 本轮恰好 2 条消息
✅ 历史 NULL 行未被回填        ✅ 非 NULL 只多出本轮 assistant 那一行
✅ 探针会话/消息/Redis 令牌已清干净（残留 0）
```

> 口径说明：第一版脚本把"全表 NULL 数不变"当断言，跑出 8 → 9 判失败——那是**我断言写错**：
> 本轮新增的 **user** 行本来就该是 NULL。已改为"排除探针会话的历史 NULL 数"口径，即 §1.4 结果。

---

## 2. ② 案例类标题边界注释（仅注释，不改行为）

`backend/app/chat/citation_check.py`：

```diff
 # 改表前先想清楚：加词 = 以它结尾的一切《…》都进比对；删词 = 真法规静默漏判，两者都要配回归测试。
+# 案例材料（如『最高法发布劳动争议典型案例』）不属于法规名，不参与清单外比对；实测 b29 轮影响 0。
 LAW_NAME_SUFFIXES: tuple[str, ...] = (
```

**证据**：`citation_check.py` 改动前后 **AST 完全一致**（注释不进入 AST）——即行为逐字不变，
`_is_law_name_candidate` 判定结果与改前相同（`道路交通事故认定书` → 排除，`民法典` → 候选）。

---

## 3. 验收汇总

| 验收项 | 要求 | 实测 |
| --- | --- | --- |
| 落库 model 有值并贴出 | 真实 SSE 一轮 | ✅ `id=44 role=assistant model='deepseek-flash'` |
| 历史 NULL 行不动 | 不回填、不猜测 | ✅ 8 → 8；非 NULL 仅 +1 |
| 全量测试 | passed 数不减 | ✅ **671 passed**（665 + 6），3 warnings |
| check_services | exit 0 | ✅ exit 0（默认模式与 `--with-api` 均 0） |
| 零逻辑改动（注释类） | AST 一致 | ✅ citation_check AST 完全一致；chat_store 去 docstring 后一致 |

改动文件 md5 与行数：

| 文件 | md5 | 行数 |
| --- | --- | --- |
| `backend/app/api/chat_persistence.py` | `5fed85eb6fef85df7648695e5130f0dc` | 243 |
| `backend/app/chat/chat_store.py` | `d18ccc12484ff835a3ab2b3fb02b445b` | 279 |
| `backend/app/chat/citation_check.py` | `3309e0df9e6512b6168a7a6e38ce14fc` | **300** |
| `backend/tests/test_chat_persistence.py` | `21eaea46618e01c2bcb2827fdb9f3bb6` | 145 |

全部文件 CRLF=0（未用文本模式写盘）；`backend/`、`evaluation/` 下无临时脚本；8124 端口已释放。

---

## 4. 需要你知道 / 待裁决的四件事（都没擅自处理）

1. **`citation_check.py` 现在正好 300 行**（本批注释 +1，299 → 300），仍满足上限但**零余量**。
   下一批抽取 `guard.resolve_citation_outcome` 前按你的前置要求拆文件时，顺带把行数降到 ≤250 即可。
2. **`check_services.py` 的 LLM 探针有盲点**（既有问题，非本批引入）：探针用 `max_tokens=16` 问
   "回复两个字：正常"，本次实测拿到 **0 字 content** → 打 ❌，但**空 content 不计入 `failures`**，
   所以退出码仍是 0（见 `b31_check_services_with_api.txt`）。即"check_services exit 0"**不能证明 LLM 可用**；
   本批 LLM 是否可用由 §1.4 的 760 字真实回答证明。
3. **历史 8 行 `model=NULL` 保持原样**（无法回溯，按你的裁决不回填、不猜测）。若日后要补，只能人工确认
   当时用的模型再写——`chat_messages` 里没有其它字段能反推模型。
4. **探针数据已清理**：脚本按固定 `user_id=e2e_b31_model_probe` 幂等删除会话与消息、并清掉 Redis 令牌键，
   跑完残留 0 行。若你想自己再验一次，脚本副本在 `reports/b31_sse_model_accept.py.txt`（原件在
   `%TEMP%/b31_bak/`），它会自起自停后端、跑完自动清理。

---

## 5. 证据文件清单

| 文件 | 内容 |
| --- | --- |
| `reports/b31_sse_model_accept.txt` | 真实 SSE 落库验收 10/10（含落库原始行） |
| `reports/b31_sse_model_accept.py.txt` | 验收脚本副本（自起自停、幂等清理） |
| `reports/b31_reverse_proof.txt` | 反向证明（旧行为 2 failed → 字节级还原） |
| `reports/b31_pytest.txt` | 全量测试 671 passed |
| `reports/b31_check_services.txt` | 服务自检（默认模式）exit 0 |
| `reports/b31_check_services_with_api.txt` | 服务自检（`--with-api`）exit 0，含 §4.2 的 LLM 探针 ❌ |
