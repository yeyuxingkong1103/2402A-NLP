# SF6 评测集生成 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 基于《GB/T 44653-2024 六氟化硫气体现场循环再利用导则》PDF，生成 3 个独立 JSON 评测集文件：`general_eval.json`、`safety_eval.json`、`retrieval_eval.json`。

**架构：** 先从 PDF 抽取章节结构、关键条款、数值阈值和高风险要求，再按用途拆成三套数据：通用理解、风险防护、检索定位。所有题目都必须能回指到原文页码与证据片段；检索集额外加入明确的反样本，避免把正文外内容误当作答案。

**技术栈：** `pypdf`、Python `json`、本地文件系统、现有 `eval/sets/` 目录约定。

---

### 任务 1：确定 PDF 文本抽取口径

**文件：**
- 创建：`scripts/extract_sf6_eval_source.py`
- 测试：`backend/tests/test_pdf_eval_extraction.py`

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path
from scripts.extract_sf6_eval_source import extract_pages


def test_extract_pages_returns_page_text():
    pdf_path = Path(r"C:\Users\tirito\Downloads\GB_T_44653-2024_六氟化硫气体现场循环再利用导则.pdf")
    pages = extract_pages(pdf_path)

    assert len(pages) >= 10
    assert pages[0].page_number == 1
    assert "六氟化硫" in pages[0].text
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest backend/tests/test_pdf_eval_extraction.py -v`
预期：FAIL，提示 `extract_pages` 尚未实现。

- [ ] **步骤 3：编写最少实现代码**

```python
from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader


@dataclass(frozen=True)
class PdfPageText:
    page_number: int
    text: str


def extract_pages(pdf_path: Path) -> list[PdfPageText]:
    reader = PdfReader(str(pdf_path))
    pages: list[PdfPageText] = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(PdfPageText(page_number=index, text=page.extract_text() or ""))
    return pages
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest backend/tests/test_pdf_eval_extraction.py -v`
预期：PASS，至少能稳定取出页码与正文文本。

- [ ] **步骤 5：Commit**

```bash
git add scripts/extract_sf6_eval_source.py backend/tests/test_pdf_eval_extraction.py
git commit -m "feat: add pdf text extraction helper"
```

### 任务 2：定义评测集 JSON 结构与输出约定

**文件：**
- 创建：`scripts/build_sf6_eval_sets.py`
- 测试：`backend/tests/test_sf6_eval_schema.py`

- [ ] **步骤 1：编写失败的测试**

```python
from scripts.build_sf6_eval_sets import EvalItem, EvalSet


def test_eval_item_has_required_fields():
    item = EvalItem(
        id="retrieval-01",
        question="标准适用于什么场景？",
        answer="适用于电气设备中 SF6 气体的现场循环再利用。",
        label="positive",
        source_pages=[7],
        evidence="本文件适用于电气设备中 SF6 气体的现场循环再利用",
        topic="scope",
        difficulty="easy",
        notes="",
    )

    assert item.id == "retrieval-01"
    assert item.label == "positive"


def test_eval_set_serializes_to_json_list():
    eval_set = EvalSet(name="general_eval", items=[])

    assert eval_set.to_json() == []
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest backend/tests/test_sf6_eval_schema.py -v`
预期：FAIL，提示 `EvalItem`、`EvalSet` 尚未定义。

- [ ] **步骤 3：编写最少实现代码**

```python
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    answer: str
    label: str
    source_pages: list[int]
    evidence: str
    topic: str
    difficulty: str
    notes: str


@dataclass(frozen=True)
class EvalSet:
    name: str
    items: list[EvalItem]

    def to_json(self) -> list[dict]:
        return [asdict(item) for item in self.items]
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest backend/tests/test_sf6_eval_schema.py -v`
预期：PASS，确保后续三个 JSON 都能用同一结构输出。

- [ ] **步骤 5：Commit**

```bash
git add scripts/build_sf6_eval_sets.py backend/tests/test_sf6_eval_schema.py
git commit -m "feat: define sf6 eval set schema"
```

### 任务 3：生成通用评测集

**文件：**
- 修改：`scripts/build_sf6_eval_sets.py`
- 测试：`backend/tests/test_sf6_general_eval.py`
- 输出：`eval/sets/general_eval.json`

- [ ] **步骤 1：编写失败的测试**

```python
import json
from pathlib import Path
from scripts.build_sf6_eval_sets import build_general_eval


def test_general_eval_has_about_ten_items(tmp_path: Path):
    output_path = tmp_path / "general_eval.json"
    build_general_eval(output_path)
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert 8 <= len(data) <= 12
    assert all(item["label"] == "positive" for item in data)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest backend/tests/test_sf6_general_eval.py -v`
预期：FAIL，因为生成器函数尚未实现。

- [ ] **步骤 3：编写最少实现代码**

```python
def build_general_eval(output_path):
    items = [
        EvalItem(
            id="general-01",
            question="这份标准主要规定了什么流程？",
            answer="规定了 SF6 气体的现场检测、回收、净化、回充全过程循环再利用方法。",
            label="positive",
            source_pages=[7],
            evidence="规定了 SF6 气体的现场检测、回收、净化、回充的技术指标及安全防护措施",
            topic="scope",
            difficulty="easy",
            notes="",
        ),
    ]
    output_path.write_text(json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest backend/tests/test_sf6_general_eval.py -v`
预期：PASS，随后补足到 8–12 条并覆盖章节分布。

- [ ] **步骤 5：Commit**

```bash
git add scripts/build_sf6_eval_sets.py backend/tests/test_sf6_general_eval.py eval/sets/general_eval.json
git commit -m "feat: add general sf6 eval set"
```

### 任务 4：生成安全评测集

**文件：**
- 修改：`scripts/build_sf6_eval_sets.py`
- 测试：`backend/tests/test_sf6_safety_eval.py`
- 输出：`eval/sets/safety_eval.json`

- [ ] **步骤 1：编写失败的测试**

```python
import json
from pathlib import Path
from scripts.build_sf6_eval_sets import build_safety_eval


def test_safety_eval_has_five_items(tmp_path: Path):
    output_path = tmp_path / "safety_eval.json"
    build_safety_eval(output_path)
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(data) == 5
    assert all(item["topic"] in {"safety", "protection", "limit"} for item in data)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest backend/tests/test_sf6_safety_eval.py -v`
预期：FAIL，因为安全集生成器尚未实现。

- [ ] **步骤 3：编写最少实现代码**

```python
def build_safety_eval(output_path):
    items = [
        EvalItem(
            id="safety-01",
            question="使用过的 SF6 储气容器在存储和使用时应注意什么？",
            answer="应关紧阀门、戴上瓶帽，防止剩余气体泄漏。",
            label="positive",
            source_pages=[15],
            evidence="使用过的 SF6 气体储气容器应关紧阀门，戴上瓶帽，应防止剩余气体泄漏",
            topic="safety",
            difficulty="easy",
            notes="",
        )
    ]
    output_path.write_text(json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest backend/tests/test_sf6_safety_eval.py -v`
预期：PASS，再补充到 5 条，覆盖第 11 章的防护要求和关键阈值。

- [ ] **步骤 5：Commit**

```bash
git add scripts/build_sf6_eval_sets.py backend/tests/test_sf6_safety_eval.py eval/sets/safety_eval.json
git commit -m "feat: add sf6 safety eval set"
```

### 任务 5：生成检索评测集

**文件：**
- 修改：`scripts/build_sf6_eval_sets.py`
- 测试：`backend/tests/test_sf6_retrieval_eval.py`
- 输出：`eval/sets/retrieval_eval.json`

- [ ] **步骤 1：编写失败的测试**

```python
import json
from pathlib import Path
from scripts.build_sf6_eval_sets import build_retrieval_eval


def test_retrieval_eval_ratio_and_negatives(tmp_path: Path):
    output_path = tmp_path / "retrieval_eval.json"
    build_retrieval_eval(output_path)
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(data) == 15
    positives = [item for item in data if item["label"] == "positive"]
    negatives = [item for item in data if item["label"] == "negative"]
    assert len(positives) == 11
    assert len(negatives) == 4
    assert all(item["evidence"] for item in data)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest backend/tests/test_sf6_retrieval_eval.py -v`
预期：FAIL，因为检索集生成器尚未实现。

- [ ] **步骤 3：编写最少实现代码**

```python
def build_retrieval_eval(output_path):
    items = [
        EvalItem(
            id="retrieval-01",
            question="这份标准适用于哪些对象？",
            answer="适用于电气设备中 SF6 气体的现场循环再利用。",
            label="positive",
            source_pages=[7],
            evidence="本文件适用于电气设备中 SF6 气体的现场循环再利用",
            topic="scope",
            difficulty="easy",
            notes="",
        ),
        EvalItem(
            id="retrieval-12",
            question="标准是否规定了氢气泄漏检测方法？",
            answer="未规定。",
            label="negative",
            source_pages=[],
            evidence="正文未出现氢气泄漏检测相关规定。",
            topic="out_of_scope",
            difficulty="medium",
            notes="反例：正文外内容",
        ),
    ]
    output_path.write_text(json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest backend/tests/test_sf6_retrieval_eval.py -v`
预期：PASS，再补齐到 15 条，并确保 11/4 比例、正反样本区分和“正文不存在”的反例都成立。

- [ ] **步骤 5：Commit**

```bash
git add scripts/build_sf6_eval_sets.py backend/tests/test_sf6_retrieval_eval.py eval/sets/retrieval_eval.json
git commit -m "feat: add sf6 retrieval eval set"
```

### 任务 6：补齐评测集目录说明与生成入口

**文件：**
- 修改：`eval/sets/README.md`
- 修改：`README.md` 或 `scripts/README.md`（若需要生成入口说明）
- 测试：`backend/tests/test_docs.py`

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path


def test_eval_sets_readme_mentions_json_outputs():
    text = Path("eval/sets/README.md").read_text(encoding="utf-8")
    assert "general_eval.json" in text
    assert "safety_eval.json" in text
    assert "retrieval_eval.json" in text
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest backend/tests/test_docs.py -v`
预期：FAIL，直到 README 明确写出三个独立 JSON 的约定。

- [ ] **步骤 3：编写最少实现代码**

```markdown
# eval/sets

本目录存放真实评测集 JSON。

- `general_eval.json`：通用评测集
- `safety_eval.json`：安全评测集
- `retrieval_eval.json`：检索评测集

所有问题必须来源于真实 PDF 内容或真实用户查询，不允许编造。
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest backend/tests/test_docs.py -v`
预期：PASS，说明目录约定已经与三份 JSON 对齐。

- [ ] **步骤 5：Commit**

```bash
git add eval/sets/README.md README.md scripts/README.md backend/tests/test_docs.py
git commit -m "docs: document sf6 eval set outputs"
```

### 任务 7：全量生成与验收

**文件：**
- 生成：`eval/sets/general_eval.json`
- 生成：`eval/sets/safety_eval.json`
- 生成：`eval/sets/retrieval_eval.json`
- 测试：`backend/tests/test_sf6_general_eval.py`
- 测试：`backend/tests/test_sf6_safety_eval.py`
- 测试：`backend/tests/test_sf6_retrieval_eval.py`
- 测试：`backend/tests/test_pdf_eval_extraction.py`

- [ ] **步骤 1：执行全部评测集测试**

运行：`pytest backend/tests/test_pdf_eval_extraction.py backend/tests/test_sf6_general_eval.py backend/tests/test_sf6_safety_eval.py backend/tests/test_sf6_retrieval_eval.py -v`
预期：PASS。

- [ ] **步骤 2：人工抽查 JSON 内容**

运行：`python -c "import json, pathlib; [print(p, len(json.loads(pathlib.Path(p).read_text(encoding='utf-8')))) for p in ['eval/sets/general_eval.json', 'eval/sets/safety_eval.json', 'eval/sets/retrieval_eval.json']]"`
预期：输出三份文件及题量，且分别接近 10、5、15。

- [ ] **步骤 3：Commit**

```bash
git add eval/sets/general_eval.json eval/sets/safety_eval.json eval/sets/retrieval_eval.json scripts/build_sf6_eval_sets.py scripts/extract_sf6_eval_source.py backend/tests/test_pdf_eval_extraction.py backend/tests/test_sf6_general_eval.py backend/tests/test_sf6_safety_eval.py backend/tests/test_sf6_retrieval_eval.py eval/sets/README.md
git commit -m "feat: generate sf6 eval datasets"
```

## 自检结果

- 规格覆盖度：已覆盖 3 个 JSON 文件的生成、结构、比例约束、反样本要求、目录说明和最终验收。
- 占位符扫描：未使用“待定”“TODO”“后续实现”等禁用表述；每个步骤都给了明确的命令与预期。
- 类型一致性：`EvalItem` / `EvalSet` / `PdfPageText` 在后续任务中保持一致，输出字段统一为 `id`、`question`、`answer`、`label`、`source_pages`、`evidence`、`topic`、`difficulty`、`notes`。
