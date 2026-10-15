# Quickstart: 验证 MinerU PDF 解析

**Feature**: `specs/001-mineru-pdf-parse` | **Date**: 2026-09-22

本文件是**验证指南**，不是实现文档。实现见 `backend/parse_pdf.py`，契约见 [contracts/cli.md](./contracts/cli.md)。

---

## 前置条件

| 项 | 要求 | 现状（2026-09-22 实测） |
|---|---|---|
| 受支持运行时 | `D:/zg6_Project/9/med_rag/rag/python.exe` | ✅ 存在，Python 3.12.14 |
| `mineru` 包 | 已安装进 `rag/` | ❌ **未安装**，需按 `requirements.txt` 手动装 |
| pipeline 权重 | 本地完整副本 | ⚠️ C: 有完整副本（2.1 G），E: 副本缺 4 个子目录 |
| vlm 权重 | 本地副本 | ✅ C: 与 E: 各有完整副本（2.2 G） |
| 语料 | `data/source_data/*.pdf` | ✅ 1 份，15 页 |

> ⚠️ **本机无 NVIDIA 显卡**（仅 Intel Iris Xe 核显）。安装 PyTorch 时**选 CPU 版**，否则会白下 2.5 GB 且无法运行。

---

## 步骤 1：安装依赖（人工执行）

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m pip install -r D:/zg6_Project/9/med_rag/requirements.txt
```

若 `mineru` 的依赖解析拉入 CUDA 版 PyTorch，改用 CPU 版索引：

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 步骤 2：确认安装成功

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -c "import mineru; print(mineru.__version__)"
```

**预期**：打印一个 `3.x.y` 版本号。
**实测后必须做**：把该版本号回填到 `requirements.txt`、`docs/04 §12` 的 `mineru_version`、以及 `specs/001-mineru-pdf-parse/research.md` 的 R1。

---

## 步骤 3：干跑（不调用 MinerU）

```bash
D:/zg6_Project/9/med_rag/rag/python.exe backend/parse_pdf.py --dry-run
```

**预期**：打印将处理的 PDF、解析出的 `doc_id`、将使用的后端与权重路径，**不**创建 `data/parsed/`，**不**调用 MinerU。

**这一步验证**：参数解析、`doc_id` 计算、权重路径存在性检查、配置文件路径推断。

---

## 步骤 4：正式解析

```bash
D:/zg6_Project/9/med_rag/rag/python.exe backend/parse_pdf.py
```

**预期**：退出码 `0`，打印类型分布报告（见 `contracts/cli.md` §5）。

**对应规格验收项**：US1 场景 1–3、SC-001（≤10 分钟）、SC-004（必然产出类型分布报告）。

---

## 步骤 5：验证产物（核心断言）

### 5.1 产物存在

```bash
ls D:/zg6_Project/9/med_rag/data/parsed/
```

**预期**：一个以 12 位十六进制命名的目录。

### 5.2 页码链路完整（最重要的断言）

用受支持运行时执行：

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -c "
import json, pathlib
d = sorted(pathlib.Path('D:/zg6_Project/9/med_rag/data/parsed').iterdir())[-1]
items = json.loads((d / 'content_list.json').read_text(encoding='utf-8'))
missing = [i for i, x in enumerate(items) if 'page_idx' not in x]
print('块总数:', len(items))
print('缺 page_idx 的块:', missing)
print('page_idx 范围:', min(x['page_idx'] for x in items), '-', max(x['page_idx'] for x in items))
"
```

**预期**：`缺 page_idx 的块: []`，`page_idx 范围: 0 - 14`（与 PDF 实际 15 页一致）。
**若页码缺失**：对应 `docs/04 §15` 待验证项 **P2**，须修正页码换算逻辑并回写 `docs/02 §3`。

### 5.3 类型契约核对（回填文档用）

```bash
D:/zg6_Project/9/med_rag/rag/python.exe -c "
import json, pathlib, collections
d = sorted(pathlib.Path('D:/zg6_Project/9/med_rag/data/parsed').iterdir())[-1]
items = json.loads((d / 'content_list.json').read_text(encoding='utf-8'))
for t, n in collections.Counter(x['type'] for x in items).most_common():
    print(f'{t:20s} {n}')
"
```

**预期**：类型取值全部落在 `data-model.md` §5.5 的表内。
**若出现表外类型**：**停止并人工决策**，回写 `docs/04 §5` 的类型表——这是 `docs/04 §15` 的待验证项 **P1**，也是本特性最需要人工确认的一步。

### 5.4 与文档实测值对照

`docs/04 §3` 给出的实测基线：

| 属性 | 文档记录值 | 你的实测值 |
|---|---|---|
| 页数 | 15 | ? |
| 中文字符数 | 17,996 | ? |

字符数可由 `content_list.json` 中 `type == "text"` 的 `text` 拼接后统计。**显著偏离需查明原因**。

---

## 步骤 6：验证幂等跳过（SC-003）

```bash
D:/zg6_Project/9/med_rag/rag/python.exe backend/parse_pdf.py
```

**预期**：同一文档 `status=skipped`，耗时相对首次下降 90% 以上，产物内容逐字节不变。

再验证强制重解析：

```bash
D:/zg6_Project/9/med_rag/rag/python.exe backend/parse_pdf.py --force
```

**预期**：忽略已有产物重新解析。

---

## 步骤 7：验证失败路径（SC-005）

逐条验证 `spec.md` Edge Cases 中至少这几条，**每条都应非 0 退出且不留下不完整产物**：

| 输入 | 预期退出码 |
|---|---|
| 不存在的路径 | `1` |
| 非 PDF 文件（如 `.txt`） | `1` |
| 权重路径指到不存在的目录 | `1` |
| 人为把 `content_list.json` 删掉后重跑 | 应重新解析（不视为已完成） |

---

## 步骤 8（可选）：验证 `vlm-engine`

仅在你想确认 `docs/04 §12` 是否该改用 vlm 时执行。**预期可能失败**——官方对 Windows + 纯 CPU 的支持矩阵标为不支持。

```bash
D:/zg6_Project/9/med_rag/rag/python.exe backend/parse_pdf.py --backend vlm-engine --force
```

**结果无论成败都要记入** `research.md` 的 R3，以及 `docs/04 §15`。

---

## 附带效果：需要回填的文档

验证完成后，以下位置需要人工更新（本特性不自动修改 docs）：

| 位置 | 回填内容 |
|---|---|
| `requirements.txt` | `mineru` 的精确版本号 |
| `docs/04 §12` | `mineru_version` 实测量；`mineru_backend` 由 `pipeline` 改为"默认 pipeline，可切 vlm-engine" |
| `docs/04 §5` | 类型表的实测核对结果（P1） |
| `docs/04 §4` | 补记 manifest 准入闸门被跳过的偏离（见 `plan.md` Complexity Tracking） |
| `docs/04 §15` | P1、P2 的验证结果 |
| `specs/001-mineru-pdf-parse/research.md` | R1–R5 的实测结论 |
