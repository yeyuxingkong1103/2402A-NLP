# 设计：拆分 backend/parse_pdf.py

日期：2026-09-22
状态：已评审通过，待实现

## 问题

`backend/parse_pdf.py` 338 行单文件，包含四个互不相关的关注点：输入发现、MinerU
外部依赖调用、产物定位与校验、命令行编排。读任何一个都要跨过另外三个。

## 目标

按职责拆成 4 个文件，但**运行方式一个字都不变**：

```
rag/python.exe backend/parse_pdf.py [INPUT] [选项]
```

命令、选项、退出码（0/1/2/3）全部保持现状。这样 `specs/001-mineru-pdf-parse/`
下的 `contracts/cli.md` 与 `quickstart.md` 中的 8 条验证命令继续有效，无需改动。

## 非目标

- 不改 `docs/05 §5` 定义的 `-m backend.pipeline <子命令>` 接口形态。那是 S3–S6
  全部落地时才需要推进的事，本次只解决可读性。
- 不改任何解析行为、默认值、校验规则或产物路径。
- 不引入测试框架。
- 不同步更新 `specs/` 中"单文件脚本"的描述性文字（见文末"已知遗留"）。

## 文件布局

```
backend/
  parse_pdf.py          入口（约 130 行）
  parse/
    __init__.py         共享常量与 ScriptError（约 40 行）
    mineru_env.py       MinerU 外部依赖（约 95 行）
    artifacts.py        产物定位与校验（约 110 行）
```

## 单元职责与接口

划分判据：**"要不要 MinerU 装好才能跑"**决定归属 `mineru_env.py`；
**"要不要磁盘上有产物才能跑"**决定归属 `artifacts.py`。据此 `artifacts.py`
可以脱开 MinerU 独立执行——glob 修复的验证正是这么做的。

### `parse/__init__.py`

只放被多方共享、且不属于任何一方的定义，不含逻辑，因此无循环导入风险。

- `ScriptError`（携带退出码）
- `EXIT_OK` / `EXIT_ARG` / `EXIT_VALIDATION` / `EXIT_DEPENDENCY`
- `REPO_ROOT`、`SOURCE_DIR`、`PARSED_DIR`、`DEFAULT_CONFIG_PATH`
- `DEFAULT_PIPELINE_MODELS`、`DEFAULT_VLM_MODELS`
- `BACKENDS`、`DEFAULT_BACKEND`

### `parse/mineru_env.py`

对外的全部纯函数，不读写产物目录：

`write_mineru_config`、`build_mineru_env`、`check_weights`、
`resolve_mineru_exe`、`probe_mineru_version`、`run_mineru`

依赖：`parse`（常量、`ScriptError`）、标准库。

### `parse/artifacts.py`

对磁盘产物的全部操作，不调用 MinerU：

`locate_content_list`、`load_content_items`、`validate_page_idx`、
`is_artifact_complete`、`build_type_distribution`、`print_report`、
模块常量 `KNOWN_BLOCK_TYPES`

依赖：`parse`、标准库。

### `backend/parse_pdf.py`

输入发现（`compute_doc_id`、`discover_documents`）、编排（`process_one`）、
命令行（`parse_args`、`main`）。import 上面两个模块，反向不被 import。

## 关键约束：REPO_ROOT 的推断

现状 `parse_pdf.py:42`：

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
```

文件若挪进 `backend/parse/`，`parent.parent` 会从 `med_rag/` 变成 `backend/`，
脚本将**静默地**把产物写到 `backend\data\parsed\`——不报错，只是找不着。这是本次
拆分最易踩且最难察觉的坑。

处理：`REPO_ROOT` 只在 `parse/__init__.py` 中计算一次，用**三层** `parent`
（`parse/__init__.py` → `parse` → `backend` → `med_rag`），使其只依赖包自身位置，
与谁 import 它无关。`MED_RAG_REPO_ROOT` 环境变量的覆盖优先级保持不变。

## 导入方式

`python backend\parse_pdf.py` 运行时 `sys.path[0]` 为 `backend\`，
`from parse import ...` 可命中；但 `-m backend.parse_pdf` 会失效。入口两种都支持：

```python
try:
    from .parse import artifacts, mineru_env
except ImportError:
    from parse import artifacts, mineru_env
```

`parse/` 内部模块之间一律用相对导入（`from . import REPO_ROOT`），
使包被认作 `parse` 或 `backend.parse` 时均成立。

## 验证方案

无测试框架，采用改动前后对拍：

1. **拆分动工前**重新采一次 `--dry-run` 基线，重定向到文件。必须显式设
   `PYTHONIOENCODING=utf-8`——控制台默认走 GBK，中文输出会变乱码，
   拿去 diff 得不出可信结论：

   ```bash
   PYTHONIOENCODING=utf-8 rag/python.exe backend/parse_pdf.py --dry-run > /tmp/before.txt 2>&1
   ```

2. 拆分完成后用同样命令生成 `after.txt`，逐字 diff。两者都应包含
   `后端 pipeline（权重 …PDF-Extract-Kit-1.0…）`、`待处理 1 份`、
   `doc_id=d6da41b5d356` 三行关键信息
3. 直接 import 模块依次调用 `artifacts.locate_content_list` →
   `artifacts.load_content_items` → `artifacts.validate_page_idx` →
   `artifacts.build_type_distribution`，与基线比对：

   ```
   块数 304
   {text: 257, header: 19, page_number: 15, table: 8,
    image: 2, footer: 1, list: 1, page_footnote: 1}
   ```

4. 跑完整命令，确认 `skipped` 且退出码为 0

## 已知遗留

`specs/001-mineru-pdf-parse/` 中有四处描述本次改动后会变陈旧，按决定**本次不改**：

| 位置 | 内容 |
|---|---|
| `plan.md:11` | "交付物是一个单文件脚本" |
| `plan.md:76` | 目录树只列了 `parse_pdf.py` |
| `tasks.md:7` | "单文件" |
| `tasks.md:49` | T-20 行数核对 250–350 行 |

`contracts/cli.md`（调用命令、退出码）与 `quickstart.md`（8 条验证命令）不受影响。

## 实施后补充：模块级 API 的收缩（未在上文记录）

改动前，`backend/parse_pdf.py` 顶层暴露全部 17 个函数与常量（`locate_content_list`、
`KNOWN_BLOCK_TYPES`、`EXIT_*`、`run_mineru` …）。拆分后入口只剩 5 个函数，
其余必须经由 `parse_pdf.artifacts.*` / `parse_pdf.mineru_env.*` 访问。

仓库内无受害者（已 grep 确认无运行时代码 import `parse_pdf`），但仓库**外**的临时脚本会踩空——
例如生成 `baseline-artifacts.txt` 的那段脚本用的 `import parse_pdf as P; P.locate_content_list(...)`
现在会 `AttributeError`。若将来有 notebook 或运维脚本按旧名调用，需改走子模块。

### 实施后补充：`-m` 调用方式的回归与修复

入口最初写成单一的 `from parse import ...`，导致 `python -m backend.parse_pdf` 从可用变为
`ModuleNotFoundError`。现已恢复本节「导入方式」原本给出的 `try/except` 双分支写法，
两种调用方式（文件形式与 `-m`）均与基线逐字对拍通过。

（过程中曾以「`backend/` 无 `__init__.py` 故 `-m` 不成立」为由删掉该分支——该判断在 Python 3
不成立，隐式命名空间包使 `-m` 可以导入。）
