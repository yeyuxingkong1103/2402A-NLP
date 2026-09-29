# scripts/e2e/ —— 真实链路端到端验收脚本

与 `tests/` 的区别：这里的脚本**起真后端进程 + 连真 MySQL/Milvus/Redis + 调线上模型 API**，
用于"整条链路是否真的通了"的验收（注册落库、SSE 流式、审核发布、长期记忆等）。
`tests/` 里的单测/集成测试用 SQLite + 假客户端，两者互补，不可互相替代。

## 统一入口（批次 37 起）

```bash
python scripts/e2e/run_integration_tests.py                 # 全部 5 项（认证落库/长期记忆/审核发布/PDF解析/多轮+摘要）
python scripts/e2e/run_integration_tests.py --skip-pdf      # 跳过 PDF（省 MinerU 额度）
python scripts/e2e/run_integration_tests.py --only auth,memory
```

逐项失败不阻断，跑完输出总表与总体退出码（0=全过）；收尾自动清理探针账号
（`cleanup_e2e_probe_accounts.py`），逐项日志落 `%TEMP%/integration_tests/<时间戳>/`。

## 共同前置

| 项 | 要求 |
| --- | --- |
| 配置 | 项目根 `.env` 齐全（脚本经 `scripts/_env.py` 读取，源码内不含任何口令） |
| 依赖服务 | MySQL / Redis / Milvus 在线；可用 `python scripts/check_services.py` 先自检（全绿再跑） |
| 模型 API | Embedding / Reranker / LLM 线上可用（`check_services.py --with-api` 验证） |
| 后端环境 | `ENVIRONMENT=test`（注册/改密走**集成测试旁路固定码 `000000`**；见下节） |
| 端口 | 8123（刻意避开本机被 Docker Desktop 占用的 8000） |
| 解释器 | `C:/Users/92842/anaconda3/python.exe`（脚本内 `PY` 常量） |

运行方式（在项目根执行）：

```bash
python scripts/e2e/e2e_auth_persist.py
```

## 验证码：集成测试旁路固定码（批次 38）

`/api/v1/auth/register/code` 与 `/reset/code` 已按生产口径**不回显验证码**、统一走 SMTP 真发，
而集成测试起独立后端进程，既读不到进程内的验证码、也不便收信。因此加了**仅 test 环境**的旁路：

| 环境（`ENVIRONMENT`） | 固定码 `000000` | 真实验证码 | 旁路 INFO 日志 |
| --- | --- | --- | --- |
| `test` | ✅ 放行 | ✅ 照常校验（一次性消费） | ✅ 出现 |
| `development` | ❌ 拒绝 | ✅ 照常校验 | ❌ 不出现 |
| `production` | ❌ 拒绝 | ✅ 照常校验 | ❌ 不出现 |

- 实现：`app/auth/service.py` 的 `AuthService._consume_code()`（register / reset 共用的唯一校验入口），
  进入条件先判 `settings.environment == "test"`，非 test 环境不可达该分支；
- 守卫测试：`backend/tests/test_auth_test_bypass.py`（含 production/development 拒绝固定码、日志缺席断言）；
- 探针脚本自行注入 `ENVIRONMENT=test`（只注入给后端子进程），业务断言不受影响
  （注册落库、令牌有效、跨用户 404、审核留痕都照旧验真）。

## 脚本清单

| 脚本 | 作用 | 额外前置 | 可重复运行 | 运行后清理 |
| --- | --- | --- | --- | --- |
| `e2e_auth_persist.py` | 批次5：注册 → **真实杀进程重启** → 登录成功 → SSE 问答 → 会话列表/历史 → 跨用户 404 → password_hash 非明文 | 无（自己起停后端） | ✅ 是（邮箱与会话名带 8 位随机后缀） | 留下 2 个 `e2e_auth_persist_*@qq.com` 账号 → 跑 `cleanup_e2e_probe_accounts.py` |
| `e2e_create_admin.py` | 批次7-1：`create_admin` CLI 四项（未注册报错 / 升级为管理员 / 管理员接口 200 / 重复执行幂等） | 无（自己起后端） | ✅ 是（邮箱带随机后缀） | 留下 1 个 `e2e_admin_cli_*@qq.com`（**管理员**）→ 跑清理脚本 |
| `e2e_review_publish.py` | 阶段6：未登录 401 / 普通用户 403；版本置回 pending → 检索不到 → approve 后检索到；reject → 向量减少；审核留痕 | 迁移已完成（11 版本 approved） | ⚠️ 是，但**会真实改动知识库**（doc 9 置 pending、doc 6 驳回），末尾有收尾恢复（重置状态 + 重新索引） | 中途失败会留脏状态：按收尾段的两条 SQL + `index_legal_documents` 手工恢复；账号跑清理脚本 |
| `e2e_long_term_memory.py` | 批次14：长期记忆 写入 → 检索 → 注入提示词 → 删除 → 开关（跨用户隔离 / 去重） | 无 | ⚠️ **谨慎**：`USER_A/USER_B/SESSION` 是固定值（`e2ea…`/`e2eb…`/`e2e_session_mem`），记忆写入真实 Milvus 且软删除仅置位 `deleted`，重复跑会与历史残留相互干扰 | Milvus 里的 `e2ea…`/`e2eb…` 记忆**没有对应 users 行**，清理脚本覆盖不到，需手工按 `user_id` 删除或 drop 后重建集合 |
| `cleanup_e2e_probe_accounts.py` | 批次15：按死条件（`password_hash` 非 48 字节 urlsafe-base64 **或** `email LIKE 'e2e%@%'`）删探针账号及其从属数据；Milvus 按实体 `delete`，不 drop 集合 | 无 | ✅ 是（幂等：无匹配行则不删任何东西） | 本身就是清理工具；保留真实用户数据不动 |
| `pdf_parse_main_path.py` | 批次22：真实 PDF 跑 **MinerU 主分支**（提交 → 预签名上传 → 轮询 → 下载 zip 取 `full.md`）→ 清洗 → 切块，并打印 batch_id/轮询状态/条号序列/章节标题存活 | 远程 MinerU 可用（**消耗额度**）；样本在 `data/pdf_samples/` | ✅ 是（无状态；产物只落 `--out-dir`） | 产物全在 `--out-dir`（默认系统临时目录，脚本会打印路径）：`*.mineru_raw.md` / `*.cleaned.txt` / `*.summary.json` —— 整目录删除即可；不写库、不写 Redis |
| `pdf_parse_qwen_fallback.py` | 批次22：**假 MinerU**（不联网、人为注入 `complete=False`）触发兜底 + **真实 Qwen-VL** 渲染页图 OCR → 兜底分支正文/切块验收 | Qwen-VL（DashScope 兼容模式）可用（**消耗额度**） | ✅ 是（同上） | `--out-dir` 内 `*.qwen_raw.txt` / `*.qwen_cleaned.txt` / `*.qwen_summary.json`，整目录删除即可 |
| `compare_pdf_vs_html.py` | 批次22：同一部法规「PDF 解析 vs 库内 HTML 解析」对照 —— 条号序列、逐条正文（规范化后）、款/项子块数、结构性差异（目录 / Markdown 残留） | 库内有对应 approved 文档（默认 `documents#9`）；默认现跑一次真实 MinerU，加 `--cleaned 文件` 则复用产物、**完全不联网** | ✅ 是（只读库：`SELECT` 而已，不写任何数据） | 用 `--cleaned` 时零产物；不走 `--cleaned` 时会在系统临时目录留一份 `*.cleaned.txt`（可删除） |
| `fetch_pdf_samples.py` | 批次23：从**原始政府网站 URL** 重下 `data/pdf_samples/` 的 4 份回归样本，并用 PyMuPDF 校验页数与文字层是否与 `README` 清单一致（站点改版/换附件会立刻暴露） | `data/pdf_samples/` 可写；默认走网络（加 `--verify-only` 则完全不联网） | ✅ 是（幂等：已存在则跳过下载；`--force` 强制重下） | **无副作用**：只写 PDF；产物就是样本本身，不需要清理 |

#### `fetch_pdf_samples.py` 用法与退出码

```bash
# 缺哪份下哪份，然后 4 份全部校验（页数 + 文字层）
python scripts/e2e/fetch_pdf_samples.py

# 全部重下（怀疑本地样本被改过时用）
python scripts/e2e/fetch_pdf_samples.py --force

# 只校验，不联网（离线/无外网时用）
python scripts/e2e/fetch_pdf_samples.py --verify-only
```

`0` 全部通过；`1` 有样本与预期不符（页数/文字层超容差）；`2` 前置不满足
（PyMuPDF 未装 / 目录不存在 / 样本缺失）。预期值写在脚本的 `TARGETS` 常量里，
**与 `data/pdf_samples/README.md` 的清单表逐字对应，改一处必须改两处**。

### 批次 22 新增的三个 PDF 脚本（补充说明）

- **为什么放在 e2e 而不是 tests**：它们要"真的打远程 MinerU / 真的做 Qwen-VL OCR"，
  单测里全用替身（`tests/test_mineru_client.py`、`tests/test_qwen_vl_client.py`）；
  两边互补——这里验"外部服务真的能用"，单测验"我们的代码逻辑对"。
- **退出码约定**（三个脚本一致）：`0` 通过；`1` 链路不符合预期（MinerU 不完整 / 没走兜底分支 /
  条号序列不一致）；`2` 前置不满足（PDF 不存在、缺配置）。
- **可重复性**：都对同一份 `data/pdf_samples/` 样本**无副作用**（不写库、不写 Redis）。
  唯一成本是模型额度：主路径 1 次 MinerU、兜底 1 次 Qwen-VL（10 页 ≈ 3 批）、
  对照默认 1 次 MinerU。
- **已知差异不算失败**：`compare_pdf_vs_html.py` 报出「3 条法条的子块数不一致」是
  **已登记差异**（PDF 版面换行被还原成 `\n\n`，同一"款"多切 1 个子块；条级正文 100% 一致），
  脚本会就地打印说明；详见 `reports/batch22_pdf_pipeline_acceptance.md` 第四节。

## 账号与数据清理约定

1. 所有 e2e 账号邮箱都以 `e2e` 开头，这是清理脚本的**唯一识别依据之一**；
   新写 e2e 脚本时请沿用该前缀，否则清理脚本扫不到。
2. 清理脚本只在 `scripts/e2e/` 下维护这一份，不要在别处再写一份删除逻辑。
3. 清理前后都会打印行数核对（users / chat_sessions / chat_messages / Milvus 实体），
   以"目标 user_key 残留 = 0"作为通过标准。

## 已知限制（未修，需要时再排期）

- `e2e_auth_persist.py` / `e2e_create_admin.py` / `e2e_review_publish.py` 里
  `BACKEND_DIR`、`PY`、`RAG` 仍是本机绝对路径（`C:\Users\92842\...`），换机器需改这三处；
  改为由 `Path(__file__).resolve().parents[2]` 推导是小改动，但要先确认不影响
  "后端子进程 cwd"的语义。
- `e2e_long_term_memory.py` 的固定 `user_id` 是它不可重复运行的根因（见上表）。
