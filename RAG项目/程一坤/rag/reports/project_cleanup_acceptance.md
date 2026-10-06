# 项目垃圾清理 · 验收报告

> 执行日期：2026-09-21 ｜ 依据：`桌面/法律RAG-代码与垃圾清理处理文档.md`
> 可逆性：全部删除内容已先备份至 `%TEMP%/clean_bak/`（reports 全量 285 文件 + demo_evidence + 3 个数据备份目录 + 代码改前副本 + 清单）

## 一、执行结果总览

| 类别 | 内容 | 结果 |
|---|---|---|
| 一 缓存/构建产物 | `__pycache__`×16、`.pytest_cache`、`frontend/.next`（157MB）、空目录 `frontend/src/components/legal/` | ✅ 清除；`frontend/public/` 保留 |
| 二 可重建数据备份 | `labor_law_processed_backup_*` 3 个（7.5MB） | ✅ 删除；重建验证通过（见下） |
| 四 reports 瘦身 | 285 文件/31MB → **233 文件/24MB**（净删 52 文件/约 8MB，另含误删回补 20 件） | ✅ 逐文件引用核对 |
| 五 演示取证 | `demo_evidence/`（1.3MB，16 文件） | ✅ 删除（验收报告/docs 零引用） |
| 七 代码去重 | `chat_endpoint()` 两处逐字重复 → 抽到 `http_retry.chat_completions_endpoint` | ✅ + 6 条锁定测试 |
| 六 工具目录 | `.idea` / `.claude` / `.superpowers` / `.workbuddy` | ⏸ 未动，待用户拍板 |

## 二、重建验证（第二类）

`run_offline_ingest --output-root <临时目录>` 从 raw HTML 重建：**success=11 / failure=0**，包数量 11 == 11。
差异说明：crawl_records/manifest 为抓取时间戳；3 个包的 chunks/versions 因切分规则演进 version_id 不同（472681 差 1 块）——当前库与当前包一致（check_services 绿），旧备份是更早快照，删除安全。

## 三、reports 引用核对方法（第四类）

- 保留：109 份 `.md` 报告 + 全部 `latest_*` 基线（同族被 `_final`/`_r3` 取代的 6 个除外）+ 逐名引用的证据文件 + 20 个**模式引用**文件（如 `synonym_ablation_{off,on}_{smoke,guard,pass1,pass2,pass3}.json`、`eval_20260920_152111_batch11.*`、迁移回滚记录 `law_status_vocabulary_migration_<ts>.json`——此为 `--rollback` 功能依赖，必须保留）。
- 初轮自动核对删除 72 件后，模式引用复查**回补 20 件**；复检确认每个留存文件均可归类（md / latest_* / 被点名 / 模式引用）。

## 四、代码改动（第七类）

| 文件 | 改动 | 行数 |
|---|---|---|
| `app/models/http_retry.py` | +`chat_completions_endpoint()` 共用实现 | 247→256 |
| `app/models/llm.py` | 属性改为委托 + import | 248→249 |
| `app/models/qwen_vl.py` | 属性改为委托（import 并入既有行） | 299→299 |
| `tests/test_chat_endpoint_shared.py` | 新增 6 条锁定用例（行为不变 + 委托事实） | 新文件 |

Protocol 双声明（vector_index_service/assembly 的 `embed()`）按文档**未合并**；`config_defaults.py`、PyMuPDF 两处误报**未动**。

## 五、终验收（全绿）

| 项 | 期望 | 实测 |
|---|---|---|
| 全量测试 | 646 passed | ✅ **652 passed**（646 + 6 新增锁定用例） |
| check_services | exit 0 | ✅ exit 0 |
| check_imports | exit 0 | ✅ exit 0（106 文件） |
| demo_ask 端到端 | 带引用正常回答 | ✅ 5 条法源引用 + 护栏通过 |
| `npm run build` | 可重建 | ✅ 构建成功（验证后 .next 再次清除） |
| 目录终检 | 白名单 | ✅ 零 `__pycache__` / 零 `.pytest_cache` / 零 `.next`；backend 根严格白名单（含 `.env.example`） |

## 六、体积变化

- 项目总大小：清理前约 560MB → **395MB**（主要是 `.next` 157MB + 缓存 + reports/备份/demo_evidence 约 18MB）
- 未动的大头：`frontend/node_modules`（依赖，非垃圾）

**报告结束** ｜ 2026-09-21
