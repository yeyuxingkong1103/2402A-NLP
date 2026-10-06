# 分块 v3 设计：结构化条款重组（过滤噪声 + 标题正文合并 + 条款号元数据）

> **日期**：2026-09-03 ｜ **状态**：草案待审查 ｜ **语料**：以 GB/T 44653-2024《六氟化硫气体现场循环再利用导则》为主
> **方案**：chunking 前新增结构化重组层（方案 A），产出"条款单元流"；恢复 `MinerUBlock.bbox` 做版面页眉/页码通用判定
> **前置澄清**：①纯分块层、不动清洗（前页整页剔除 / ASCII 化 / VLM 裁剪均不在本轮，需另立设计）②"行号噪声"=孤立条款号短块 ③标题与正文按叶子条款合并 ④条款元数据存"当前条款归属号" ⑤页眉按版面位置判定（非关键词）

---

## 1. 现状与问题（实测，195 chunks 脏库 p1–p15）

当前分块 = `build_chunks` 直接吃 `MinerUBlock[]`，每块 800 字符硬切。真实数据暴露的问题：

| # | 问题 | 实测证据 |
|---|---|---|
| 1 | **孤立条款号短块** | `3.1`/`3.2`/`3.3`/`3.4` 单独成块（label 混为 text/title），无正文，污染检索 |
| 2 | **页眉混入** | P7 页首 `六氟化硫（SF6）气体的 / 现场循环再利用导则`（应是每页页眉，本篇多数页 MinerU 未抽出，仅 P7 出现） |
| 3 | **标题与正文分离** | `[title] 5.4.6 SF6气体分解产物检测装置` 与其正文 `[text] SF 气体分解产物检测装置技术参数应满足…` 是两个独立块、独立 chunk |
| 4 | **条款号无元数据** | 号分布杂乱：独立成块（`3.1`）/ 在 title 内（`5.4.6 …`）/ 在 text 开头（`5.6.1 检测管路…`、`8.4.1 …`），无法按条款定位 |

**真实结构规律**（p7–p9 块序）：
- 模式 A：条款号拆成独立短块（术语段 `3.2` + 英文定义行 + 定义正文，三段式）
- 模式 B：`[title 含条款号]` 紧跟 1 个 `[text]` 正文（绝大多数条款，两段式）
- 模式 C：父/子 title 连排无 text（`5.4`→`5.4.1`→…，父条款无独立正文）

## 2. 目标

对分块环节做一次结构化改造，产出可按条款检索、可追溯、无版面噪声的 chunk：

1. **过滤**孤立条款号短块（不再单独成 chunk）与版面页眉/页码；
2. **标题与正文合并**为叶子条款单元；
3. **条款归属号写入元数据**（`Chunk.clause_id`）；
4. 超长条款仍按 800 续切，但标题挂在首块、clause_id 写满所有续片，保证按条款可全量召回。

**红线（宪法）**：只改构建产物与分块逻辑；不改 `page`（物理页，评测 `expected_page` 口径不动）；不引入清洗类规则（ASCII/前页/VLM）；不动检索/重排/问答链路。

## 3. 非目标（后置）

- 前页（封面/目次/前言）整页剔除——不在本轮（上次 v3 清洗已回滚，需另立设计）
- 数字上下标 ASCII 化（SF→SF6）、⁃ 修复、VLM 表/图/公式裁剪——均不属本分块轮
- 引用关系提取（把正文中引用的 DL/T/GB 号结构化）——只存当前条款归属号
- 语义分块 / 父子分块策略升级

## 4. 架构与数据流

chunking 前插入结构化重组层：

```
MinerUBlock[]（有序，含 bbox）
   ─▶ [structure.py 结构化重组]
        ① 版面页眉/页码过滤（按 bbox 位置 + 短文本）
        ② 条款流构建：title含号→开新叶；孤立号块→并入后续 title；text/table/image→归当前条款
        ③ 归属传播与叶子判定
   ─▶ ClauseUnit[]（clause_id / title / title_page / blocks）
   ─▶ build_chunks：每叶条款→chunk；超长800续切、clause_id写入每片
   ─▶ Chunk[]（含 clause_id）→ 向量化入库
```

**关键约束**：`page` 仍为 PDF 物理页；本次不重建既有库，真实验收时再重建评测。

## 5. 模块划分与组件

| 文件 | 改动 | 职责 |
|---|---|---|
| `backend/app/mineru.py` | 改 | 恢复 `MinerUBlock.bbox: list[float]\|None` + middle.json 提取（仅坐标） |
| `backend/app/structure.py` | 新增 | `build_clause_units(blocks) -> list[ClauseUnit]`：版面过滤 + 条款构建 + 归属传播（纯函数） |
| `backend/app/chunking.py` | 改 | `build_chunks` 消费 `ClauseUnit[]`；超长续切；clause 元数据写入 |
| `backend/app/models.py` | 改 | `Chunk` 增加 `clause_id: str \| None = None` |
| `backend/app/pipeline.py` | 改 | 唯一生产调用点（`pipeline.py:63`）：先 `structure.build_clause_units(blocks)` 再 `build_chunks(document_id, units)` |

### 5.1 bbox 恢复（mineru.py）
`MinerUBlock` 加 `bbox`；`_parse_middle_json` 从块 dict `bbox` 键提取 `[x0,y0,x1,y1]`，缺失 `None`（pypdf fallback 无坐标）。**仅坐标用途，不涉清洗**。

### 5.2 结构化重组（structure.py，纯函数）
`ClauseUnit`：
```python
@dataclass
class ClauseUnit:
    clause_id: str    # 叶子条款号，如 "5.4.6"；无条款归属正文为 ""
    title: str        # 标题文本（不含条款号）
    body: str         # 拼好的正文文本（title/正文/可能 table 文本合并，供 build_chunks 直接切）
    title_page: int   # 标题所在物理页
```

- **版面页眉/页码过滤**：块 bbox 在上部 ~10%（`y1/页高 < 0.1`）且文本短（≤20 字）→ 判页眉剔除；bbox 在底部 ~8%（`y0/页高 > 0.92`）且整块纯数字 → 页码剔除。**靠版面位置 + 短文本，不做正文关键词匹配**（防误删引用清单，如 GB/T 150.2 压力容器…）。
- **条款流构建**（单遍扫描，父无正文靠"关闭时 body 空则丢弃"自动消失）：
  - **规则 A（开新条款）**：`title` 块文本匹配 `条款号 + 空白 + 标题`（如 `5.4.6 xxx`、`1 范围`、`5.4 检测装置`）→ 关闭当前单元（body 空则丢弃），开启该号新单元、title=号后文本；或任意 label 匹配**整块纯条款号**（如 `3.1`）→ 开启 title 为空的号单元（其后术语名行/定义作 body）
  - **规则 B（条号化正文归父）**：`text` 块匹配 `条款号 + 空白 + 正文`（如 `5.6.1 检测管路应…`，MinerU 把子条号与正文并入同一 text）→ 并入当前父条款 body，不开新条款
  - **规则 C（普通内容）**：其余 text / 无号 title / table / image → 并入当前单元 body；无当前单元则建 `clause_id=""`（前言/标准名前内容）
- **叶子判定**：父条款仅含子 title 无正文（如 `5.4 检测装置` 后紧跟 `5.4.1`）→ 子标题触发 A 时父 body 空被丢弃，不产父汇总单元；子条款各自成单元。
- **跨页条款**：title 在某页开启后，后续页 text 只要没有新条款标题打断，仍归入同一单元；`page` 用 title 所在页（按块遍历天然跨页连续）。
- 无条款归属正文（标准名前/前言/参考文献 `参 考 文 献`）→ clause_id=""，仍成 chunk。

### 5.3 分块消费 ClauseUnit（chunking.py）
- 输入 `ClauseUnit[]`。每叶条款组装 `title + "\n" + body`；
- ≤800 → 单 chunk；超长 → 800 续切，**标题只挂首 chunk，每个续片都写 clause_id**（可按条款全量召回）；
- `page` 用 `title_page`（无归属前言单元 title_page 取首个内容块所在页）；
- `Chunk.category` 统一为 `"text"`（单元已合并 title/正文/可能 table），不再保原始 label；追溯以 `clause_id` 为单元。

### 5.4 数据模型（models.py）
`Chunk` 加 `clause_id: str | None = None`（Qdrant payload 同步带出，供按条款检索/审计）。

## 6. 错误处理

- 条款构建遇异常 title（无法 parse 号）→ 跳过该 title 作为正文并入当前条款，不崩溃；
- 空正文单元 → 不产 chunk；
- 重组异常 → 不吞，沿 pipeline 落到任务 `failed` + `error_message`（NFR5）；
- bbox 缺失块（pypdf fallback）→ 版面过滤跳过（不误剔），仍按规则归属。

## 7. 测试

- **mineru.py**：bbox 字段与提取（带 bbox / 无 bbox / 畸形 bbox）；
- **structure.py**：合成块序 → 断言 ClauseUnit 划分、孤立号块并入、父/子 title 连排叶子判定、text 归属、无条款前言处理；
- **版面过滤**：合成带 y 坐标块 → 页眉/页码剔除、中下部正文（如引用清单 GB/T 150.2）保留；
- **chunking.py**：ClauseUnit 超长续切、clause_id 写满续片、title 挂首块；
- **models.py**：Chunk.clause_id 默认 None、可设置。

## 8. 验收标准

1. **真实重建**：删除旧库 → 用新分块管线重入库 GB/T 44653 PDF → chunk 可按 clause_id 审计：
   - 不再出现"纯条款号、无内容"的孤立 chunk（每个条款号都与其标题/正文内容同 chunk，如 `3.1` 术语号与术语名/定义合块）
   - 标题与正文同 chunk（如 5.4.6 标题与其技术参数正文在同一 clause_id 下）
   - 无页眉块（P7 顶部文档题名组不出库；引用清单正文仍在）
   - **版面阈值标定**：重建前先对真实 bbox 采样（spike），确认 y 从页顶计、用 `y0/页高` 比例标定"上部 10% / 底部 8%"阈值（页眉块 y0 应显著小于正文首行），阈值以实测为准
2. **条款元数据**：正文条款 chunk 均带非空 clause_id；按 clause_id 能取回 5.4.6 全部续片；
3. **43 题评测**：对比 v2 基线 recall/MRR/refusal，**不退化**（分块结构变化可能影响排序，须实测确认；若退化记录归因）；
4. 全部 pytest 通过。

## 9. 提交说明

> 本目录非 git 仓库（`git rev-parse` 无输出），本设计不执行 commit；若后续纳入版本管理再一并提交。v3 需求文档 / 版本迭代回写待真实验收后按宪法执行。
