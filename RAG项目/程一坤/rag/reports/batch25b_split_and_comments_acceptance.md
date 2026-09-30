# 批次 25-续（25-2）：超限文件拆分 + 3 文件补注释 —— 验收报告

任务来源：用户对批次 25 的四项裁决（② 300 行冲突分档 / ③ check_imports / ④ 清理备份）
边界遵守：未改 `docs/`；临时脚本全在 `C:/Users/92842/AppData/Local/Temp/b25_bak/`（**未进** `backend/`）；
项目根只新增 `.dev_bak25b/`（本批备份，验收后可清）。

---

## 一、结论速览

| # | 裁决 | 结果 |
|---|---|---|
| ②-a | `models/mineru.py`（446）按职责拆成「客户端 API / 上传与轮询下载流程 / 结果解析」 | ✅ 拆成 **4 个文件**，全部 ≤250 行、覆盖率 ≥30% |
| ②-a′ | `core/config.py`（381）按「默认值常量表 / 读取与校验」拆 | ✅ 拆成 **3 个文件**，全部 ≤250 行、覆盖率 ≥30% |
| ②-b | `models/qwen_vl.py`、`retrieval/keyword_search.py`、`retrieval/vector_search.py` 不拆、只补到不超 300 | ✅ 三者均 **299 行**（补注释后），未超限 |
| ②-c | 零逻辑改动（AST 逐对象比对 PASS） | ✅ 指纹脚本退出码 0；另加**运行时对拍 18/18** |
| ③ | `scripts/check_imports.py` 的 `SKIP_DIRS` 加 `.dev_bak*` | ✅ 已加；扫描数回到 **101**、退出码 0 |
| ④ | 清理 `.dev_bak25/`（已验收） | ✅ 已删（含 6 个文件，删除前列出核对） |
| — | 体检基线 | ✅ **598 passed** / `check_services` 退出码 0 / `check_imports` 退出码 0 |

---

## 二、拆分成果与行数

### 2.1 `models/mineru.py`：446 → 244（+3 个新模块）

| 文件 | 行数 | 职责 |
|---|---|---|
| `models/mineru.py` | **244** | 模块常量 + `MineruClient.__init__` + `parse()` 编排 + `_build_payload()` + `_failed()` + `build_mineru_client_from_settings()` |
| `models/mineru_api.py` | **107** | **客户端 API 调用**：`_auth_headers` / `_call`（重试包装）/ 三种默认传输实现 |
| `models/mineru_flow.py` | **153** | **上传与轮询下载流程**：`_create_batch` → `_upload_file` → `_await_result` → `_download_zip`（+ `_extract_result_entry`） |
| `models/mineru_result.py` | **149** | **结果解析**：`find_entry` / `count_pdf_pages` / `extract_markdown` / `resolve_page_count` |

组合方式：`class MineruClient(MineruFlowMixin, MineruApiMixin)` —— 用混入组合而非复制粘贴，
方法名与签名保持原样，`MineruClient._request_upload` 这类直接取方法的写法仍然可用。

**两处"刻意不搬"**（都有测试锚点，搬了会让测试静默失效，不是偷懒）：

1. `MAX_FILE_BYTES` 与读它的 `_build_payload()` 必须同模块 ——
   测试用 `monkeypatch.setattr("app.models.mineru.MAX_FILE_BYTES", 0)` 触发尺寸拦截；
   常量与读它的函数分处两个模块时，patch 只改本模块名字、读到的仍是旧值，**用例会"看起来通过"**。
2. `__init__` 不外移 —— 客户端状态的唯一初始化点，拆开会让"谁定义了哪些属性"难以追踪
   （两个 mixin 里只写了**纯注解**声明，不赋默认值 —— 赋了会把"漏初始化"从 AttributeError 变成静默默认值）。

### 2.2 `core/config.py`：381 → 215（+2 个新模块）

| 文件 | 行数 | 职责 |
|---|---|---|
| `core/config.py` | **215** | `load_environment_file()` + `Settings` 类 + `settings` 实例 + 路径常量再导出 |
| `core/config_defaults.py` | **193** | **默认值常量表**（`DEFAULT_*` 44 个 + `PRODUCTION_REQUIRED_KEYS` + 路径常量） |
| `core/config_loader.py` | **233** | **读取与校验逻辑**（`build_settings_from_environment(cls)`，即原 `from_environment` 方法体） |

`Settings` 类留在 `config.py`（按你的要求），字段默认值统一改为引用 `defaults.DEFAULT_*`；
每个取值的实测依据（如拒答阈值、记忆去重 0.70、40/40/40 窗口）**跟着常量一起搬到常量表**，
放在常量上方而不是字段上方 —— 改值时依据就在眼前。

**一处"刻意不搬"**：`load_environment_file()` 留在 `config.py`，`from_environment()` 沦为薄包装：
测试用 `monkeypatch.setattr("app.core.config.load_environment_file", lambda *a, **k: 0)` 屏蔽真实 `.env`，
被 patch 的名字必须与**调用点**同模块命名空间，否则真实 `.env` 的 `SMTP_*` 会被读回、
"生产缺配置"这个场景根本模拟不出来。

---

## 三、零逻辑改动 —— 两份独立证据

### 3.1 证据①：AST 逐对象比对（剥 docstring）

脚本 `b25b_fingerprint.py`；粒度＝模块级函数/类方法/类体字段/模块级变量；
函数再拆「参数表」与「函数体」（参数表允许"去掉首个 self"，因为方法要变模块函数）；
跨类搬迁用显式映射（`MineruClient._call → MineruApiMixin._call` 等 14 条）后再逐字比对。

| 组 | 逐字相同 | 已声明差异 | 未声明差异 | 新增对象 |
|---|---|---|---|---|
| mineru.py 拆 4 文件 | **22** | 2 | **0** | 15（2 个 mixin + 13 条纯注解） |
| config.py 拆 3 文件 | **16** | 50 | **0** | 50（44 常量 + 必填键 + loader 函数等） |
| qwen_vl.py 补注释 | **17** | 0 | **0** | 0 |
| keyword_search.py 补注释 | **19** | 0 | **0** | 0 |
| vector_search.py 补注释 | **8** | 0 | **0** | 0 |

**结论：退出码 0，全部差异均已声明且逐条给出理由**（3 个"只补注释"文件的对象数 = 100% 逐字相同）。

已声明的差异只有三类，逐个列出：

| 对象 | 差异内容 | 理由 |
|---|---|---|
| `MineruClient`（类头） | 基类 `object` → `(MineruFlowMixin, MineruApiMixin)` | 承载被搬走的 14 个方法，组合而非复制 |
| `MineruClient.parse` | 两行调用点：`self._extract_markdown(archive)` → `extract_markdown(archive)`、`self._resolve_page_count(...)` → `resolve_page_count(...)` | 函数外移后不再是方法 |
| `Settings.from_environment` | 188 行方法体 → 1 行薄包装 | 方法体逐字搬到 `config_loader`；留 `load_environment_file()` 保 monkeypatch 锚点 |
| `Settings.REQUIRED_IN_PRODUCTION` | `default=(...字面量)` → `default=defaults.PRODUCTION_REQUIRED_KEYS` | 常量提取；值由证据②证明相同 |
| 46 个 `Settings.<字段>` | 字面量 → `defaults.DEFAULT_*` | 常量提取；值由证据②证明相同 |
| `PROJECT_ROOT_DIRECTORY` / `ENVIRONMENT_FILE_PATH` | `Path(__file__)...parents[3]` → 常量表定义 + 本模块再导出 | 同目录，`parents[3]` 语义不变；值由证据②证明相同 |

### 3.2 证据②：运行时等价对拍（`b25b_runtime_dump.py`）—— **18/18 通过**

旧实现从 `.dev_bak25b/` 以独立模块名加载，与新实现**同进程、同解释器、同环境变量**跑：

| 项 | 结果 |
|---|---|
| 字段清单（名/类型/默认值）逐项相同 | ✅ **60 个字段**全等 |
| `Settings()` 实例逐字段取值相同 | ✅ |
| 每个字段的运行时类型相同 | ✅ 无差异 |
| `REQUIRED_IN_PRODUCTION` 相同 | ✅ 9 个键 |
| `PROJECT_ROOT_DIRECTORY` / `ENVIRONMENT_FILE_PATH` 再导出且值相同 | ✅ 指向项目根 |
| `Settings.from_environment` / `load_environment_file` 签名相同 | ✅ |
| `.env` 解析行为相同 | ✅ 同一份含"注释行/空行/`=` 前后空格/空值/空键/重复键"的探针 .env：写入 3 条、取值 `{'B25B_ALPHA': '1', 'B25B_BETA': '2', 'B25B_GAMMA': ''}` 两侧逐字相同 |
| `MineruClient.__init__` 签名完全相同 | ✅ 19 个参数 |
| 类方法集合 = 旧集合 − 已搬迁的 2 个 | ✅ 旧 15 − 2 = 13 = 新 13（差额恰好） |
| 模块级常量相同 | ✅ `209715200` / `200` |
| `parse()` 成功路径：返回结果 / `last_trace` / 请求序列 | ✅ 全部相同（4 次请求逐条同） |
| `parse()` 失败路径（任务 failed）：同上 | ✅ 全部相同（3 次请求逐条同） |

> 非确定性字段说明：`last_trace.elapsed_seconds` 是 wall-clock 实测耗时（两次执行必然不同），
> 对拍时剔除；剔除后其余字段逐字相同。

---

## 四、3 个文件"补到接近但不超过 300 行"

| 文件 | 行数 前 → 后 | 覆盖率 前 → 后 | 加了什么 |
|---|---|---|---|
| `models/qwen_vl.py` | 297 → **299** | 17.85% → 18.06% | `_extract_text` 里"为什么用下标而非 `.get()`"（结构不符要抛错交给上层收敛） |
| `retrieval/keyword_search.py` | 295 → **299** | 17.63% → 18.06% | `_resolve_current` 的 docstring："证据不足返回 None 而非 False，以及为什么必须区分" |
| `retrieval/vector_search.py` | 290 → **299** | 24.83% → 27.09% | 模块边界说明（filters/assembly 的分工）、`RetrievalError` 为什么要与"没找到"区分、Milvus 返回结构 `list[list[hit]]` 取 `[0]`、`output_fields` 挂在 `entity` 下、父块字段优先 |

**必须说清的一个矛盾（需要你知情）**：这三个文件**做不到覆盖率 ≥30%**，不是没补够，是数学上不可能 ——
"≤300 行"与"≥30% 注释"在这三个文件上互相排斥：

| 文件 | 当前注释行 | 要到 30% 需补 | 补完总行数 |
|---|---|---|---|
| `qwen_vl.py`（299 行） | 54 | ≥51 行 | **350 行** ❌ |
| `keyword_search.py`（299 行） | 54 | ≥51 行 | **350 行** ❌ |
| `vector_search.py`（299 行） | 81 | ≥13 行 | **312 行** ❌ |

（推导：`(注释+x)/(总行+x) ≥ 0.30`，299 行的文件每补 1 行注释同时增 1 行分母，收敛不到 30%。）

按你的裁决"不拆只补到不超限"，本批**以 300 行为硬约束优先**，覆盖率停在 18%/18%/27%。
若今后要它们也 ≥30%，只有两条路：**拆文件**（像 mineru/config 那样）或**豁免行数**。等你定。

---

## 五、`config_loader.py` 的覆盖率补齐（本批自查发现的遗漏）

`config_loader.py` 是新拆分产物，首版覆盖率只有 **21.93%**（因为原 `from_environment` 方法体本身
注释就稀），低于我们在批次 25 立的"本次拆分产生的新文件 ≥30%"门槛。已补齐到 **30.47%**：

- **加 22 行配置域说明**（每个配置域"影响什么、为什么这样读"），例如
  "向量化：换模型/维度等于换向量空间，必须同时重建索引"、
  "同义扩写只影响 BM25 那一路，向量路与重排输入不动"、
  "MinerU 未配 key 只是 PDF 解析不可用，所以不在 production 必填清单里"；
- **把 10 处多行 `os.getenv(...)` 调用压成更紧凑的换行**腾出行数 ——
  ⚠️ 这类压缩**只改书写格式、不改 AST**（参数列表尾逗号、换行位置都不进 AST），
  已由证据①的"逐字相同"数量不变 + 证据②的字段默认值全等共同证明。

结果：**233 行（≤250）+ 30.47%（≥30%）**。

---

## 六、`scripts/check_imports.py` 的改动（比你说的"1 行"多一行）

原实现把 `.dev_bak21` 硬编码在 `SKIP_DIRS` 里（精确匹配），批次 25 我建的 `.dev_bak25`
没被跳过 → 扫描数从 **101 误报成 107**。本次改为**前缀判定**，以后每批的 `.dev_bakNN` 自动跳过：

```python
# 备份目录前缀：.dev_bak21 / .dev_bak25 / .dev_bak25b … 全部跳过
BACKUP_DIR_PREFIX = ".dev_bak"
...
        if any(
            part in SKIP_DIRS or part.startswith(BACKUP_DIR_PREFIX)
            for part in path.parts
        ):
```

即：删掉硬编码项 + 加前缀判定 + 加注释，实际 **2 处**改动（你说的是"1 行"，这里如实报备）。
验证：扫描数回到 **101**、退出码 **0**。

---

## 七、覆盖率前后总账

| 指标 | 批次 25 后 | 本批后 |
|---|---|---|
| 全 `app/` 文件数 | 109 | **114**（拆分新增 5 个模块） |
| 覆盖率 <30% 的文件 | 49 | **47** |
| 任务书点名 20 个文件 | 20/20 达标 | **20/20 达标** |
| 单文件超 300 行 | 5 个（mineru 446 / config 381 / qwen_vl 297 / keyword_search 295 / vector_search 290） | **0 个** |

7 个新拆分产物的行数与覆盖率：

| 文件 | 行数 | 覆盖率 | ≤250 行 | ≥30% |
|---|---|---|---|---|
| `core/config_loader.py` | 233 | 30.47% | ✅ | ✅ |
| `core/config.py` | 215 | 43.26% | ✅ | ✅ |
| `core/config_defaults.py` | 193 | 53.89% | ✅ | ✅ |
| `models/mineru.py` | 244 | 32.79% | ✅ | ✅ |
| `models/mineru_result.py` | 149 | 42.95% | ✅ | ✅ |
| `models/mineru_flow.py` | 153 | 35.29% | ✅ | ✅ |
| `models/mineru_api.py` | 107 | 36.45% | ✅ | ✅ |

---

## 八、体检基线（真实输出）

| 项 | 命令 | 结果 |
|---|---|---|
| 单元测试 | `PYTHONPATH= <rag> -m pytest backend/tests -q` | ✅ **598 passed**, 3 warnings in 14.27s，退出码 **0** |
| 服务体检 | `python scripts/check_services.py` | ✅ 全绿，退出码 **0**（Redis PONG / MySQL 1627 / Milvus 1627==1627 / approved=11） |
| 导入自检 | `python scripts/check_imports.py` | ✅ 扫描 **101** 文件全绿，退出码 **0** |

---

## 九、复现命令

```bash
cd C:/Users/92842/Desktop/rag
PY="C:/Users/92842/anaconda3/envs/rag/python.exe"
T="C:/Users/92842/AppData/Local/Temp/b25_bak"

# 零逻辑改动取证
"$PY" "$T/b25b_fingerprint.py"      > reports/b25b_fingerprint.txt       # 退出码 0
"$PY" "$T/b25b_runtime_dump.py"     > reports/b25b_runtime_dump.txt      # 18/18，退出码 0

# 覆盖率
"$PY" "$T/count_comment_coverage.py" --json "$T/coverage_final.json"

# 体检
PYTHONPATH= "$PY" -m pytest backend/tests -q --no-header -p no:cacheprovider
"$PY" scripts/check_services.py
"$PY" scripts/check_imports.py
```

---

## 十、证据文件表

| 文件 | 内容 |
|---|---|
| `reports/b25b_fingerprint.txt` | 证据①：AST 逐对象比对（22+16+17+19+8 逐字相同、0 未声明差异） |
| `reports/b25b_runtime_dump.txt` | 证据②：运行时对拍 18/18（字段/默认值/.env 解析/签名/parse 端到端） |
| `reports/b25b_comment_coverage_final.txt` | 全 app 覆盖率终测（114 文件，47 个 <30%） |
| `reports/b25b_final_pytest.log` | 598 passed |
| `reports/b25b_check_services.txt` / `b25b_check_imports.txt` | 另两项体检原始输出 |
| `.dev_bak25b/` | 本批 5 个改动源文件的**改动前**版本（md5 已校验），验收后可清 |

---

## 十一、需要你处理的遗留项

1. **3 个 299 行文件的覆盖率**（18%/18%/27%）—— 与"≤300 行"硬冲突，等你定：拆 / 豁免 / 维持现状。
2. **`.dev_bak25b/`** 留还是清（按任务书"验收通过后清理"，你说清我就清）。
3. 批次 24 的 **8 项悬案**见 `reports/batch24_pending_decisions.md`（本批一并整理，未擅自处置）。
