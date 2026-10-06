# 分块 v3（结构化条款重组）实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。
> **注意：本项目不是 git 仓库**（`git rev-parse` 无输出），本计划所有 **Commit** 步骤一律跳过，改为步骤结尾输出"改动文件清单"即可。

**目标：** 在 chunking 前新增结构化重组层：①按版面位置过滤页眉/页码块；②过滤孤立条款号短块；③标题与正文按叶子条款合并；④条款归属号写入 `Chunk.clause_id` 元数据。

**架构：** 新增 `structure.py`（纯函数）：`filter_margin_blocks`（版面页眉/页码）+ `build_clause_units`（有序块→条款单元流）；`MinerUBlock` 恢复 `bbox`；`Chunk` 加 `clause_id`；`build_chunks` 改为消费 `ClauseUnit[]`；`pipeline.py` 唯一生产调用点先重组再分块。`page` 保持 PDF 物理页不变。

**技术栈：** Python 3.12、pytest、现有 MinerU 解析输出（`MinerUBlock`）、`data/state.json` 真实块数据。

**测试运行基座：** `D:\an\envs\mineru\python.exe -m pytest <file> -v`（项目根下执行）。单元测试用合成块（`MinerUBlock`），不依赖真实 MinerU。

**真实结构参考**（验收用，来自 195-chunks 脏库 p7–p9）：
- 页眉：P7 顶部 `六氟化硫（SF6）气体的` / `现场循环再利用导则`（应在 y0 很小）
- 孤立条款号：text/title `3.1`、`3.2`、`3.3`、`3.4`
- 术语两段式：title `3.2` → text `SF6气体净化处理 purification…`（英文定义行）→ text 定义正文
- 普通条款：title `5.4.6 SF6气体分解产物检测装置` → text `SF 气体分解产物检测装置技术参数应满足…`
- 父/子连排：title `5.4 检测装置` → title `5.4.1 …` → …（父无正文）
- text 开头带号：text `5.6.1 检测管路应使用不锈钢管…`（父 5.6 连接管路及接头标题其后）

---

### 任务 1：models.py —— Chunk 增加 clause_id

**文件：**
- 修改：`backend/app/models.py:110-118`（Chunk 模型）
- 测试：`backend/tests/test_models.py`（追加）

- [ ] **步骤 1：编写失败的测试**

```python
# 追加到 backend/tests/test_models.py
from backend.app.models import Chunk


def test_chunk_clause_id_defaults_none():
    c = Chunk(chunk_id="c1", document_id="d1", page=1, category="text",
              text="内容", source_span="page=1:block=1")
    assert c.clause_id is None


def test_chunk_clause_id_settable():
    c = Chunk(chunk_id="c1", document_id="d1", page=1, category="text",
              text="内容", source_span="page=1:block=1", clause_id="5.4.6")
    assert c.clause_id == "5.4.6"
```

- [ ] **步骤 2：运行测试确认失败**

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests/test_models.py -q`
预期：FAIL，`Chunk` 无 `clause_id` 字段。

- [ ] **步骤 3：实现**

```python
class Chunk(BaseModel):
    """可溯源文本块。"""

    chunk_id: str
    document_id: str
    page: int = Field(ge=1)
    category: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_span: str = Field(min_length=1)
    clause_id: str | None = None  # 所属叶子条款号，如 "5.4.6"；无归属为 None
```

- [ ] **步骤 4：运行测试确认通过**

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests/test_models.py -q`
预期：PASS（新 2 条 + 原有全过）

- [ ] **步骤 5：改动文件清单（无 git，不 commit）**

修改：`backend/app/models.py`、`backend/tests/test_models.py`

---

### 任务 2：mineru.py —— 恢复 bbox 字段与提取

**文件：**
- 修改：`backend/app/mineru.py:60-70`（模型）、`backend/app/mineru.py:164-196`（`_parse_middle_json`）
- 测试：`backend/tests/test_mineru_bbox.py`（新建）

- [ ] **步骤 1：编写失败的测试**

```python
# backend/tests/test_mineru_bbox.py
from backend.app.mineru import MinerUBlock, MinerUParser


def _middle_json_with_bbox():
    return {
        "pdf_info": [
            {
                "page_no": 6,
                "segmented_content_blocks": [
                    {"type": "title", "bbox": [63.0, 40.0, 300.0, 60.0],
                     "lines": [{"spans": [{"content": "1 范围"}]}]},
                    {"type": "text", "bbox": [63.0, 800.0, 300.0, 815.0],
                     "lines": [{"spans": [{"content": "本文件描述…"}]}]},
                    {"type": "text", "lines": [{"spans": [{"content": "无坐标正文"}]}]},
                ],
            }
        ]
    }


def test_extract_bbox_from_middle_json():
    parser = MinerUParser()
    blocks = parser._parse_middle_json(_middle_json_with_bbox())

    assert blocks[0].bbox == [63.0, 40.0, 300.0, 60.0]
    assert blocks[1].bbox == [63.0, 800.0, 300.0, 815.0]
    assert blocks[2].bbox is None


def test_bbox_is_optional_field():
    block = MinerUBlock(page=1, label="text", text="x", source_span="page=1:block=1")
    assert block.bbox is None
```

- [ ] **步骤 2：运行测试确认失败**

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests/test_mineru_bbox.py -q`
预期：FAIL，`MinerUBlock` 无 `bbox` 字段。

- [ ] **步骤 3：实现**

`mineru.py` 模型加字段：
```python
class MinerUBlock(BaseModel):
    """MinerU 解析出的结构化块。"""

    page: int
    label: str
    text: str
    source_span: str
    raw_text: str | None = None
    vision_text: str | None = None
    vision_model: str | None = None
    vision_applied: bool = False
    bbox: list[float] | None = None  # [x0,y0,x1,y1]，PDF 点坐标，页顶起；缺失 None
```

`_parse_middle_json` 构造处提取：
```python
bbox = block.get("bbox")
if not isinstance(bbox, list) or len(bbox) != 4:
    bbox = None
blocks.append(
    MinerUBlock(
        page=page_number,
        label=label,
        text=text,
        source_span=f"page={page_number}:block={block_index}",
        bbox=bbox,
    )
)
```

- [ ] **步骤 4：运行测试确认通过**

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests/test_mineru_bbox.py -q`
预期：PASS（2 passed）

- [ ] **步骤 5：改动文件清单（无 git，不 commit）**

修改：`backend/app/mineru.py`；新增：`backend/tests/test_mineru_bbox.py`

---

### 任务 3：structure.py —— 版面页眉/页码过滤

**文件：**
- 创建：`backend/app/structure.py`
- 测试：`backend/tests/test_structure.py`（新建）

- [ ] **步骤 1：编写失败的测试（版面过滤）**

```python
# backend/tests/test_structure.py
from backend.app.mineru import MinerUBlock
from backend.app.structure import filter_margin_blocks


def _blk(page, label, text, bbox, span):
    return MinerUBlock(page=page, label=label, text=text, source_span=span, bbox=bbox)


def test_filter_margin_blocks_drops_page_header_and_page_number():
    blocks = [
        # 页眉：页面顶部 y0 很小 + 短文本（文档题名）
        _blk(7, "title", "六氟化硫（SF6）气体的", [63, 40, 300, 60], "page=7:block=1"),
        _blk(7, "title", "现场循环再利用导则", [63, 62, 220, 80], "page=7:block=2"),
        # 正文（中部，保留）
        _blk(7, "title", "1 范围", [63, 200, 100, 215], "page=7:block=3"),
        _blk(7, "text", "本文件描述了电气设备中六氟化硫…", [63, 220, 531, 300], "page=7:block=4"),
        # 页码：页面底部 + 纯数字
        _blk(7, "text", "3", [300, 830, 320, 842], "page=7:block=9"),
    ]
    kept = filter_margin_blocks(blocks, page_height=842.0)

    texts = [b.text for b in kept]
    assert "1 范围" in texts
    assert "本文件描述了电气设备中六氟化硫…" in texts
    assert "六氟化硫（SF6）气体的" not in texts
    assert "现场循环再利用导则" not in texts
    assert "3" not in texts


def test_filter_margin_blocks_keeps_midpage_reference_list():
    # 引用清单在中部，即使文本含标准号也不剔（不做关键词匹配）
    blocks = [
        _blk(7, "text", "GB/T 150.2 压力容器 第 2 部分：材料", [63, 400, 400, 415], "page=7:block=10"),
    ]
    kept = filter_margin_blocks(blocks, page_height=842.0)
    assert [b.text for b in kept] == ["GB/T 150.2 压力容器 第 2 部分：材料"]


def test_filter_margin_blocks_keeps_block_without_bbox():
    block = MinerUBlock(page=7, label="text", text="无坐标正文",
                        source_span="page=7:block=1")
    kept = filter_margin_blocks([block], page_height=842.0)
    assert kept == [block]
```

- [ ] **步骤 2：运行测试确认失败**

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests/test_structure.py -q`
预期：FAIL，`backend.app.structure` 不存在。

- [ ] **步骤 3：实现版面过滤**

```python
# backend/app/structure.py
"""块级结构化重组：版面页眉/页码过滤 + 条款单元构建。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.mineru import MinerUBlock

# 条款号开头的标题：如 "5.4.6"、"6.1.1"、"1 范围"；捕获号与剩余标题文本
CLAUSE_HEAD_RE = re.compile(r"^\s*(?P<cid>\d+(?:\.\d+)*)(?:\s+|　+)(?P<title>.+)$")
# 整块纯条款号（孤立号块，如 "3.1"）
CLAUSE_ONLY_RE = re.compile(r"^\s*(?P<cid>\d+(?:\.\d+)*)\s*$")
# 页眉/页码文本特征（短；页眉常是文档题名/标准号）
_MARGIN_SHORT_MAX = 20


def _is_header_like_text(text: str) -> bool:
    """页眉常是短标题/标准号；仅辅助版面位置判定，绝不匹配正文关键词。"""
    return len(text.strip()) <= _MARGIN_SHORT_MAX


def filter_margin_blocks(
    blocks: list[MinerUBlock],
    page_height: float = 842.0,
    top_ratio: float = 0.10,
    bottom_ratio: float = 0.08,
) -> list[MinerUBlock]:
    """按版面位置剔除页眉（页顶）与页码（页底）。bbox 缺失的块保留。"""
    kept: list[MinerUBlock] = []
    for b in blocks:
        if b.bbox is None:
            kept.append(b)
            continue
        x0, y0, x1, y1 = b.bbox
        text = (b.text or "").strip()
        if page_height and y1 <= page_height * top_ratio and _is_header_like_text(text):
            # 页眉：位于页面上部比例内且为短文本
            continue
        if page_height and y0 >= page_height * (1 - bottom_ratio) and re.fullmatch(r"\d{1,4}", text):
            # 页码：位于页面底部且整块为纯数字
            continue
        kept.append(b)
    return kept
```

- [ ] **步骤 4：运行测试确认通过**

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests/test_structure.py -q`
预期：PASS（3 passed）

- [ ] **步骤 5：改动文件清单（无 git，不 commit）**

新增：`backend/app/structure.py`、`backend/tests/test_structure.py`

---

### 任务 4：structure.py —— 条款单元构建（ClauseUnit）

**文件：**
- 修改：`backend/app/structure.py`
- 测试：`backend/tests/test_structure.py`（追加）

- [ ] **步骤 1：追加失败的测试（条款构建）**

追加用例（蓝本取自真实块序 p7–p8）：

```python
# 追加到 backend/tests/test_structure.py
from backend.app.structure import ClauseUnit, build_clause_units


def _b(page, label, text, span, bbox=None):
    return MinerUBlock(page=page, label=label, text=text, source_span=span, bbox=bbox)


def test_build_clause_units_merges_title_and_body():
    blocks = [
        _b(9, "title", "5.4.6 SF6气体分解产物检测装置", "page=9:block=1"),
        _b(9, "text", "SF 气体分解产物检测装置技术参数应满足 DL/T 1205—2013。", "page=9:block=2"),
        _b(9, "title", "5.4.7 SF6气体纯度检测装置", "page=9:block=3"),
        _b(9, "text", "SF6 气体纯度检测装置技术参数应满足 T/CEC 140。", "page=9:block=4"),
    ]
    units = build_clause_units(blocks)

    assert [u.clause_id for u in units] == ["5.4.6", "5.4.7"]
    assert units[0].title == "SF6气体分解产物检测装置"
    assert "技术参数应满足 DL/T 1205" in units[0].body
    assert units[0].title_page == 9


def test_build_clause_units_merges_isolated_number_block_into_next_title():
    # 孤立号块 3.1 后接术语名行（中文+英文）+ 定义正文，同属条款 3.1
    blocks = [
        _b(7, "text", "3.1", "page=7:block=24"),
        _b(7, "text", "SF6 气体回收 recovery of sulfur hexafluoride gas", "page=7:block=25"),
        _b(7, "text", "将电气设备中 SF 气体采用专用装置抽出…", "page=7:block=26"),
    ]
    units = build_clause_units(blocks)

    assert len(units) == 1
    assert units[0].clause_id == "3.1"
    # 纯号块无跟随标题文本 → title 空；术语名行 + 定义都归 body
    assert units[0].title == ""
    assert "recovery of sulfur" in units[0].body
    assert "采用专用装置抽出" in units[0].body


def test_build_clause_units_parent_without_body_not_emitted():
    # 父条款 5.4 无正文，只有子条款 5.4.1 -> 不产 5.4 单元
    blocks = [
        _b(8, "title", "5.4 检测装置", "page=8:block=15"),
        _b(8, "title", "5.4.1 SF6气体中空气检测装置", "page=8:block=16"),
        _b(8, "text", "技术参数应满足 DL/T 920 中第 3 章的要求。", "page=8:block=17"),
    ]
    units = build_clause_units(blocks)

    assert [u.clause_id for u in units] == ["5.4.1"]
    assert units[0].title == "SF6气体中空气检测装置"


def test_build_clause_units_text_with_leading_clause_number_keeps_body():
    # text 以条款号开头（5.6.1）在父标题之后 -> 归父条款还是开子？按真实意图归父 5.6 下
    blocks = [
        _b(9, "title", "5.6 连接管路及接头", "page=9:block=1"),
        _b(9, "text", "5.6.1 检测管路应使用不锈钢管，内径 2 mm~4 mm。", "page=9:block=2"),
        _b(9, "text", "5.6.2 回收连接管路宜使用聚四氟乙烯管。", "page=9:block=3"),
    ]
    units = build_clause_units(blocks)

    # 5.6.1/5.6.2 是父 5.6 的条号化正文，不是独立叶子标题（无独立 title 文本后接新正文）
    assert [u.clause_id for u in units] == ["5.6"]
    assert "5.6.1 检测管路应使用不锈钢管" in units[0].body


def test_build_clause_units_preface_without_clause_goes_to_empty():
    blocks = [
        _b(7, "text", "本文件适用于电气设备中 SF 气体的现场循环再利用。", "page=7:block=4"),
    ]
    units = build_clause_units(blocks)
    assert len(units) == 1
    assert units[0].clause_id == ""
    assert units[0].body.startswith("本文件适用于")
```

- [ ] **步骤 2：运行测试确认失败**

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests/test_structure.py -q`
预期：FAIL，`ClauseUnit`/`build_clause_units` 未定义。

- [ ] **步骤 3：实现条款构建**

追加到 `structure.py`：

```python
@dataclass
class ClauseUnit:
    """一个叶子条款单元（标题 + 合并正文）。"""

    clause_id: str      # 归属条款号，如 "5.4.6"；无归属正文为 ""
    title: str          # 标题文本（不含条款号）
    body: str           # 合并正文文本（可含 table/image 文本）
    title_page: int     # 标题所在物理页
```

实现规则（纯函数，单遍扫描；无需比较条款深度——父无正文靠"关闭时 body 为空则丢弃"自动消失）：

```python
def build_clause_units(
    blocks: list[MinerUBlock],
    page_height: float = 842.0,
    top_ratio: float = 0.10,
    bottom_ratio: float = 0.08,
) -> list[ClauseUnit]:
    """版面过滤后，把有序块重组为条款单元。

    判定（每块一次，自上而下命中即 continue）：
    A. 条款标题块 = 匹配 CLAUSE_HEAD 的 title 块，或匹配 CLAUSE_ONLY（纯条款号）
       的 title/text 块（如 "5.4.6 xxx"、"5.4 检测装置"、"3.1"）：
        → 关闭当前单元（body 空则丢弃），开启该号的新单元
    B. text 块匹配 CLAUSE_HEAD（"号 + 正文"一体，如 "5.6.1 检测管路应…"）：
        → 条号化正文，并入当前父条款 body（不是新条款）
    C. 其余 text / 无号 title / table / image → 并入当前单元 body（无当前则建 clause_id=""）

    关键点：父条款（如 5.4 仅含子标题无正文）在子标题触发 A 时，因 body 空被丢弃，
    不会产出父汇总单元；子条款各自成单元。
    """
    kept = filter_margin_blocks(blocks, page_height, top_ratio, bottom_ratio)
    units: list[ClauseUnit] = []
    cur: ClauseUnit | None = None

    def _close() -> None:
        nonlocal cur
        if cur is not None and cur.body.strip():
            units.append(cur)
        cur = None

    def _new(cid: str, title: str, page: int) -> None:
        nonlocal cur
        cur = ClauseUnit(clause_id=cid, title=title, body="", title_page=page)

    for b in kept:
        text = (b.text or "").strip()
        if not text:
            continue
        label = str(b.label or "").strip().lower()

        # A. 条款标题块：title+（号+标题文本）或（title/text+纯号）
        m_head = CLAUSE_HEAD_RE.match(text)
        m_only = CLAUSE_ONLY_RE.match(text)
        if m_head and label == "title":
            _close()
            _new(m_head.group("cid"), m_head.group("title").strip(), b.page)
            continue
        if m_only:
            _close()
            _new(m_only.group("cid"), "", b.page)
            continue

        # B. text 带"号+正文"一体（5.6.1 …）：条号化正文归父
        if m_head and label == "text":
            if cur is None:
                _new("", "", b.page)
            cur.body = (cur.body + "\n" + text).strip()
            continue

        # C. 其余内容并入当前单元
        if cur is None:
            _new("", "", b.page)
        cur.body = (cur.body + "\n" + text).strip()

    _close()
    return units
```

> **算法说明**：
> - 孤立条款号块（`3.1`/`3.2`…，实测 title/text 皆有）→ 命中 A（纯号），开启 title 为空的号单元；其后术语名行 + 定义正文都是无号 text → 归 body，合并为一块。✅ 测试 `..._isolated_number_block_...` 通过。
> - title 带"号+标题"（`5.4.6 xxx`）→ 开条款；其后正文 text（无号）归 body。✅
> - 父/子连排：`5.4 检测装置`（父）开条款但无正文 → 遇子 `5.4.1 xxx` 触发 A 关闭父时 body 空被丢弃 → 只产 `5.4.1`。✅ 测试 `..._parent_without_body_...` 通过。
> - text 带"号+正文"（`5.6.1 检测管路应…`）→ 命中 B，归入父 `5.6` 的 body（`5.6` 由前面的 title 块开好）。✅ 测试 `..._text_with_leading_clause_number_keeps_body` 通过。
> - 边界：`5.6 连接管路及接头` 这类"父标题自身无独立正文、其内容全在 5.6.1~5.6.3 条号化正文行"的条款——因 5.6.1 等是 B 归父，父 5.6 会正常保留成块。若真实重建审计发现某父条款被误并/误拆，在任务 8 依真实块序微调 A/B 边界并回写本计划。

- [ ] **步骤 4：运行测试确认通过**

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests/test_structure.py -q`
预期：PASS（原 3 条 + 新 5 条，共 8 passed）

- [ ] **步骤 5：改动文件清单（无 git，不 commit）**

修改：`backend/app/structure.py`、`backend/tests/test_structure.py`

---

### 任务 5：chunking.py —— 消费 ClauseUnit、写 clause_id

**文件：**
- 修改：`backend/app/chunking.py`
- 测试：`backend/tests/test_chunking.py`（追加）

- [ ] **步骤 1：追加失败的测试**

```python
# 追加到 backend/tests/test_chunking.py
from backend.app.structure import ClauseUnit


def _unit(cid, title, body, page=9):
    return ClauseUnit(clause_id=cid, title=title, body=body, title_page=page)


def test_build_chunks_from_clause_unit_writes_clause_id():
    u = _unit("5.4.6", "SF6气体分解产物检测装置", "技术参数应满足 DL/T 1205。")
    chunks = build_chunks("doc-1", [u])
    assert len(chunks) == 1
    assert chunks[0].clause_id == "5.4.6"
    assert "SF6气体分解产物检测装置" in chunks[0].text
    assert "技术参数应满足 DL/T 1205" in chunks[0].text
    assert chunks[0].page == 9


def test_build_chunks_long_clause_splits_and_keeps_clause_id_on_all():
    u = _unit("8.3.1", "储气容器", "长" * 900)
    chunks = build_chunks("doc-1", [u], max_chars=800)
    assert len(chunks) >= 2
    # 标题只挂首块；每个续片都带 clause_id，可按条款全量召回
    assert "储气容器" in chunks[0].text
    assert all("储气容器" not in c.text for c in chunks[1:])
    assert all(c.clause_id == "8.3.1" for c in chunks)


def test_build_chunks_empty_clause_unit_produces_nothing():
    u = _unit("", "", "")
    chunks = build_chunks("doc-1", [u])
    assert chunks == []
```

- [ ] **步骤 2：运行测试确认失败**

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests/test_chunking.py -q`
预期：FAIL，`build_chunks` 仍吃 `MinerUBlock[]` 且无 clause_id。

- [ ] **步骤 3：实现**

`chunking.py` 重写 `build_chunks` 消费 `ClauseUnit`：

```python
from backend.app.structure import ClauseUnit


def build_chunks(document_id: str, units: list[ClauseUnit], max_chars: int = 800) -> list[Chunk]:
    """从条款单元构建可溯源 chunk；超长续切，clause_id 写满每片。"""
    chunks: list[Chunk] = []
    for unit in units:
        assembled = (unit.title + "\n" + unit.body).strip()
        if not assembled:
            continue
        page = unit.title_page
        if page < 1:
            raise ValueError("chunk 缺少有效页码，禁止入库")
        cleaned = clean_text(assembled)
        if not cleaned:
            continue
        parts = [cleaned[start : start + max_chars] for start in range(0, len(cleaned), max_chars)]
        for part in parts:
            chunks.append(
                Chunk(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    page=page,
                    category="text",
                    text=part,
                    source_span=unit.clause_id or "",
                    clause_id=unit.clause_id or None,
                )
            )
    return chunks
```

> **注意**：`Chunk.source_span` 现填 clause_id（单元粒度无块级 span）；`category` 统一 "text"（单元合并了 title/正文/可能 table）。若你希望保留原始块类别/span 追溯，需在 `ClauseUnit` 里带源块信息——本轮按规格以条款为追溯单元，用 clause_id 即可，不改此设计。

- [ ] **步骤 4：运行测试确认通过**

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests/test_chunking.py backend/tests/test_models.py -q`
预期：PASS（新增用例 + 原有通过）

> **旧测试改写（明确）**：`test_chunking.py` 原有 3 个用例，`build_chunks` 签名改后需同步：
> - `test_clean_text_removes_extra_space`：仍测 `clean_text`，**不改**。
> - `test_build_chunks_keeps_page_and_label` 与 `test_build_chunks_rejects_missing_page`：改传 `ClauseUnit`（不再传 `MinerUBlock`）。改写后如下——`keeps_page_and_label` 改为验证「两个 clause 单元产两 chunk、各带自身 clause_id」；`rejects_missing_page` 改为传 `title_page=0` 的单元，期望 `raise ValueError`：

```python
# backend/tests/test_chunking.py 改写两个旧用例
from backend.app.structure import ClauseUnit


def test_build_chunks_keeps_each_clause_id():
    units = [
        ClauseUnit(clause_id="3.1", title="", body="回收定义。", title_page=2),
        ClauseUnit(clause_id="3.2", title="净化", body="净化定义。", title_page=2),
    ]
    chunks = build_chunks("doc-1", units, max_chars=100)
    assert [c.clause_id for c in chunks] == ["3.1", "3.2"]
    assert [c.page for c in chunks] == [2, 2]
    assert "净化" in chunks[1].text


def test_build_chunks_rejects_missing_page():
    units = [ClauseUnit(clause_id="x", title="t", body="内容", title_page=0)]
    with pytest.raises(ValueError, match="页码"):
        build_chunks("doc-1", units, max_chars=100)
```
> 移除旧版里不再使用的 `MinerUBlock` import（若其它用例仍用则保留）。

- [ ] **步骤 5：改动文件清单（无 git，不 commit）**

修改：`backend/app/chunking.py`、`backend/tests/test_chunking.py`

---

### 任务 6：pipeline.py —— 唯一生产调用点先重组再分块

**文件：**
- 修改：`backend/app/pipeline.py:62-64`
- 测试：`backend/tests/test_pipeline.py`（更新）

- [ ] **步骤 1：更新 pipeline 调用**

`pipeline.py` **顶部 import 区**（现 `line 9 from backend.app.storage import JsonStateStore` 附近，按字母序插在 `storage` 之后）加：

```python
from backend.app.structure import build_clause_units
```

`run()` 的 CHUNKING 段（现 `pipeline.py:62-64`）改为：

```python
self.store.save_task(task.mark_running(BuildStep.CHUNKING, 45))
units = build_clause_units(blocks)
chunks = build_chunks(document.document_id, units)
self.store.save_chunks(chunks)
```

> `build_clause_units` 内部已先做版面过滤；顺序仍为 parse → `post_processor`(VLM，若配) → structure → chunking，`blocks` 传 post_processor 处理后的块即可。

- [ ] **步骤 2：更新 test_pipeline.py 断言**

`test_pipeline.py` 两个用例的 `FakeParser` 返回单块 `MinerUBlock(label="正文", text="真实内容")`，经 structure 后无条款归属 → `clause_id=None`、`category="text"`。原断言 `chunk.category == "正文"` 改为 `"text"`，并断言 `clause_id is None`。运行：

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests/test_pipeline.py -q`
预期：PASS（pipeline 端到端：单正文块 → 1 个 clause_id=None 的 chunk）

- [ ] **步骤 3：改动文件清单（无 git，不 commit）**

修改：`backend/app/pipeline.py`、`backend/tests/test_pipeline.py`

---

### 任务 7：全量单测回归

- [ ] **步骤 1：全量回归（排除真实模型/DB）**

运行：`D:\an\envs\mineru\python.exe -m pytest backend/tests -q -k "not real and not qdrant and not frontend"`
预期：全绿。本次新增/改动影响文件：`structure.py`(新)、`mineru.py`、`models.py`、`chunking.py`、`pipeline.py` + 测试 `test_structure.py`(新)/`test_mineru_bbox.py`(新)/`test_models.py`/`test_chunking.py`/`test_pipeline.py`。

- [ ] **步骤 2：改动文件清单（无 git，不 commit）**

同任务 1–6 汇总。

---

### 任务 8：真实验收（需本机 MinerU / 评测，重活）

**文件：** 无代码改动，纯运行验证。

- [ ] **步骤 1：bbox spike 标定版面阈值**

确认无进程占用 `data/qdrant` 后，运行一次真实 MinerU 解析采样 P7 页眉块与正文首块的 bbox，标定 `top_ratio`/`bottom_ratio`（设计默认 top 0.10 / bottom 0.08，页高 842 为 A4 点）。记录真实 bbox，若默认阈值误剔正文则调整并回写本计划/规格默认值。

- [ ] **步骤 2：重建向量库**

按项目约定清库后上传 GB/T 44653 PDF 并构建（或脚本重入库）。预期 task completed。

- [ ] **步骤 3：条款元数据审计**

遍历 state.json chunks，断言：
- 正文条款 chunk 的 `clause_id` 非空；可按 `clause_id` 取回 5.4.6 全部续片；
- 不再出现"纯条款号、无内容"孤立 chunk（3.1/3.2/3.3/3.4 号已与其术语名/定义同 chunk，`clause_id="3.1"` 的 chunk 文本含术语名而非仅号）；
- 标题与其正文同 chunk（5.4.6 标题文本与其正文在同一 clause_id 下）；
- P7 顶部文档题名组（页眉）不出库；引用清单正文（GB/T 150.2…）仍在。

- [ ] **步骤 4：43 题评测对比 v2 基线**

运行：`D:\an\envs\mineru\python.exe -X utf8 scripts\run_sf6_eval.py --variant both`
预期：对比 v2 基线（recall 0.7500 / MRR 0.6528 / refusal 0.92），**四指标不退化**；若因分块结构变化导致退化，如实记录归因，不静默通过。

- [ ] **步骤 5：v3 版本化回写**

按宪法：`docs/需求说明.md` 升 v3 并补迭代表（FR5 分块描述更新）；`docs/版本迭代.md` 记录 v3；`docs/整体/01-需求分析.md` 同步分块语义。完成后向用户汇报改动清单。

---

## 执行交接

计划已完成并保存到 `docs/superpowers/plans/2026-09-03-chunking-v3-structure.md`。两种执行方式：

1. **子代理驱动（推荐）** —— 每任务调度子代理，任务间审查
2. **内联执行** —— 当前会话用 executing-plans 批量执行并设检查点

> 本项目无 git；任务中 Commit 步骤已省略。任务 8 依赖本机 MinerU 与评测，需在环境就绪时执行。
