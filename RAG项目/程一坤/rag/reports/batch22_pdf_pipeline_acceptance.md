# 批次 22 验收报告（第二组：用真实 PDF 验证 PDF 解析链路）

> 第一组（批次 21 收尾三件）见 **第十节附录**；第一～九节为第二组正文。

> 边界遵守：不改 `docs/`；key 一律不出现在代码与报告；验证用的脚本**已升格**进 `scripts/eval/`、`scripts/e2e/`（项目根不再留临时脚本，见第十一节）；
> `.pdf_probe/` 的 4 份 PDF 已按指示移入 `data/pdf_samples/` 并配 README。
> 本批实跑日期：2026-09-21。

---

## 结论速览

| 验收项 | 结论 | 关键证据 |
|---|---|---|
| a 走哪条分支 + 证据 | ✅ 两条分支都真实跑通 | MinerU：`batch_id=758de4bd…`、状态 `done`；Qwen-VL：`qwen-vl-ocr`、10 页、3 批 |
| b 解析正文 / 字数 / 页数 / 耗时 | ✅ | MinerU 12,250 字 / 27 页 / **2.5s**；Qwen-VL 4,060 字 / 10 页 / 25.9s |
| c 清洗后与切块 | ✅ | MinerU：98 条 + 229 子块（含 80 项）；Qwen-VL：38 条 + 63 子块（含 19 项） |
| d PDF vs HTML 对照 | ✅ **98/98 条逐条 100% 一致** | 见第四节（含 3 个具体差异点） |
| e 章节标题短行未被吃掉 | ✅ | 10 个标题全部 原始=清洗后；非空行 262 → 262（删 0 行） |
| f 全量测试 + check_services | ✅ | **595 passed**（485 + 110）；check_services 退出码 0。**收尾后复跑 598 passed**（+3 条 page_count 测试，见第十一节） |

**过程中发现并修复 4 个真缺陷**（都不是"纸面完成"能发现的）：
① 预签名上传 403（根因是 `Content-Type` 而不是 `Authorization`）；② 大请求体导致 Qwen-VL 丢页；
③ MinerU 的 Markdown 标题行让条文边界漏 1 条；④ cleaner 吃掉「目 录」。

---

## 一、实现清单

### 新增（代码）

| 文件 | 行数 | 职责 |
|---|---|---|
| `backend/app/models/http_retry.py` | 253 | 共用「请求 + 指数退避重试」；`request_raw` 精确控制请求头 |
| `backend/app/models/mineru.py` | 405 | `MineruClient`：提交→上传→轮询→下 zip 取 `full.md` |
| `backend/app/models/qwen_vl.py` | 305 | `QwenVlClient`：渲染页图→分批 OCR→拼接 |

### 新增（测试，共 110 条）

| 文件 | 条数 |
|---|---|
| `tests/test_http_retry.py` | 25 |
| `tests/test_mineru_client.py` | 26 |
| `tests/test_qwen_vl_client.py` | 22 |
| `tests/test_cleaner_headings.py` | 14 |
| `tests/test_offline_ingest_pdf.py` | 12 |
| `tests/test_pdf_parser.py` | 6 |
| `tests/test_pdf_config.py` | 5 |

### 修改

| 文件 | 改动 |
|---|---|
| `app/core/config.py` | 接线 14 个字段（原 13 个 → 新加 `qwen_vl_max_request_bytes`）；**不进** `REQUIRED_IN_PRODUCTION` |
| `app/ingest/offline_ingest.py` | ①`parse_document` 改为注入 mineru/qwen 客户端（原只传 1 个参数，PDF 必抛）；②`collect_raw_files` 收 `.pdf`；③`media_type` 随格式走（不再硬编码 `text/html`） |
| `app/ingest/parser.py` | PDF 分支先按 Markdown 去标记再清洗（修缺陷③） |
| `app/ingest/cleaner.py` | 结构标题白名单：`第X章/节/编/部分/篇`、`第X条`、`总则/附则/…/目录`（修缺陷④） |
| `app/ingest/title_normalizer.py` | 非 HTML 不再把二进制当 UTF-8 读 `<title>`；PDF 缺人工标题时报错并点明原因 |
| `backend/requirements.txt` | `PyMuPDF==1.28.0`（不引 pdf2image） |
| `.env` / `backend/.env.example` | 新增 `QWEN_VL_MAX_REQUEST_BYTES`；注释从「未读取」改为「已读取」 |

**实现期逐字未动（哈希比对）**：`app/models/embedding.py`、`app/models/llm.py`、`app/ingest/pdf_parser.py`

```
=== embedding.py 改动后哈希对比 ===
  app/models/embedding.py        前=f9aba585db2dfa48 后=f9aba585db2dfa48  未改动 ✔
  app/models/llm.py              前=096a160f37302826 后=096a160f37302826  未改动 ✔
  app/ingest/pdf_parser.py       前=0365a6455e13edcd 后=0365a6455e13edcd  未改动 ✔
```

> 注：`pdf_parser.py` 在本批**收尾**按裁决④做了改动（兜底分支页数回退），
> 上面这组哈希是"实现期"的状态；收尾改动内容与验收见第十一节第 3 条。
> `embedding.py` / `llm.py` 全程逐字未动。

新增依赖只此一项：`PyMuPDF==1.28.0`（`pip show` → `Version: 1.28.0`；与 base 环境同版本）。
选它的理由（用户裁决 ②）：它同时是「页图渲染」和「PDF 文字层直取」两个用途的来源，不引 pdf2image。

---

## 二、验收 a～c：两条分支的真实输出

### a) 分支证据

**主路径（MinerU）**
```
目标 PDF：jm_labor_contract_law.pdf  大小=282,029 B
[1] MinerU 原始解析（未清洗）
    batch_id      = 758de4bd-3585-4106-b9c8-6793e8309c70
    轮询次数      = 2  最终状态 = done
    状态流转      = waiting-file -> done
    upload        = 282,029 B     download_zip  = 368,675 B
    complete      = True   error=(空)
```
- 状态流转取自 `MineruClient.last_trace`（客户端逐次轮询都记了 state 与耗时）。
- 结果包内文件（实测）：`full.md` / `layout.json` / `*_content_list.json` / `*_model.json` / `*_origin.pdf`，
  **不含页图** —— 所以页数不能只看页图数量。

**兜底分支（Qwen-VL，R2 口径）**
```
Qwen-VL base=https://dashscope.aliyuncs.com/compatible-mode/v1
           model=qwen-vl-ocr  超时=120.0s  每请求页数上限=8
[1] Qwen-VL 渲染 + OCR（真实调用）
    渲染页数      = 10
    请求批次数    = 3
      批 1: 图 3 张 → 识别 1434 字
      批 2: 图 3 张 → 识别  945 字
      批 3: 图 4 张 → 识别 1679 字
    识别总字符数  = 4,060
[2] parse_document（假 MinerU 不完整 → Qwen-VL 兜底）
    MinerU 被调用次数 = 1
    走上分支          = qwen-vl
```
- R2 口径如实标注：**MinerU 侧是人为注入的假客户端**（`complete=False`，不联网），
  **Qwen-VL 侧是真实调用**（模型 `qwen-vl-ocr`，对真实 PDF 的真实渲染页）。
- R1（真实扫描件）未找到：本次 4 份样本全部有文字层；找过的渠道见 `data/pdf_samples/README.md` 末尾备忘。

### b) 解析正文

| | MinerU 主路径 | Qwen-VL 兜底 |
|---|---|---|
| 文档 | 劳动合同法（27 页） | 劳动合同法实施条例（10 页） |
| 正文前 300 字 | 见下 | 见下 |
| 总字符数 | **12,250**（清洗后 12,196） | **4,060**（清洗后 4,060） |
| 页数 | **27** | **10**（`parse_document` 报 0，见"已知限制"） |
| 耗时 | **2.5 s**（含上传/轮询/下载） | **25.9 s**（10 页 = 3 次 OCR 请求） |

MinerU 正文前 300 字：
```
# 中华人民共和国劳动合同法

（2007年6月29日第十届全国人民代表大会常务委员会第二十八次会议通过 根据 2012 年 12 月 28 日
第十一届全国人民代表大会常务委员会第三十次会议《关于修改<中华人民共和国劳动合同法>的决定》修正）

## 目 录

第一章 总则
第二章 劳动合同的订立
……
## 第一章 总则

第一条 为了完善劳动合同制度，明确劳动合同双方当事人的权利和义务，保护劳动者的合法权益，构
```

Qwen-VL 正文前 300 字：
```
中华人民共和国劳动合同法实施条例

（2008 年 9 月 3 日国务院第 25 次常务会议通过 2008 年 9 月 18 日中华人民共和国国务院令第 535 号
公布 自公布之日起施行）

第一章 总则

第一条 为了贯彻实施《中华人民共和国劳动合同法》 (以下简称劳动合同法)，制定本条例。

第二条 各级人民政府和县级以上人民政府劳动行政等有关部门以及工会等组织，应当采取措施，推动
劳动合同法的贯彻实施，促进劳动关系的和谐。
```

参考对照：PyMuPDF 直取文字层 —— 劳动合同法 12,375 字 / 实施条例 4,617 字。
MinerU 12,250（差 1%）与 Qwen-VL 4,060（差 12%）都在合理范围，OCR 在小字号/全角标点上更易失真。

### c) 清洗后与切块

| | MinerU 主路径 | Qwen-VL 兜底 |
|---|---|---|
| 总块数 | 327 | 101 |
| **条（父块）** | **98** | **38** |
| 款/项（子块） | 229 | 63 |
| 其中**识别到项** | **80** | **19** |
| 其中纯款 | 149 | 44 |

劳动合同法全文正好 98 条、实施条例正好 38 条 —— **切块条数与法规实际条数完全吻合**
（MinerU 侧条号序列从「第一条」连续到「第九十八条」，无缺号）。

条款识别样例（MinerU 侧）：`第十七条` 有 2 个项子块（`item_no=1/2`，对应「（一）（二）」），
`第八十四条` 4 个子块（3 款 + …），`款/项` 都带上了 `paragraph_no`。

---

## 三、验收 e：章节标题短行专查（cleaner）

```
[4] 章节标题短行存活检查（cleaner 专查）
    第一章    原始= 2 清洗后= 2  ✔        第五章    原始= 2 清洗后= 2  ✔
    第二章    原始= 2 清洗后= 2  ✔        第六章    原始= 2 清洗后= 2  ✔
    第三章    原始= 2 清洗后= 2  ✔        第七章    原始= 2 清洗后= 2  ✔
    第四章    原始= 2 清洗后= 2  ✔        第八章    原始= 2 清洗后= 2  ✔
    总则     原始= 2 清洗后= 2  ✔        附则     原始= 2 清洗后= 2  ✔

    非空行：原始 262 → 清洗后 262（删了 0 行）
```
Qwen-VL 侧同样全 ✔（`第一章`/`第二章`/`总则`/`附则` 原始=清洗后=1）。

**专查结论：修复前确实会被吃掉。** 复现与修复过程：

| 形态 | 修复前 | 修复后 |
|---|---|---|
| `第一章`（剥掉 `## ` 后剩 3 字、无标点） | 删 | 保留 |
| `总则` | 删 | 保留 |
| `目 录`（剥掉 `## ` 后 3 字） | 删 | 保留 |
| `第一章 总则`（7 字） | 保留（≥6 字侥幸） | 保留 |

修复方式：`cleaner._is_navigation_line` 增加「特征 0」——先判是不是法规结构标题，
是就**不参与导航判定**（而不是放宽短行阈值，避免把真实导航噪音也放进来）。
白名单正则：
```python
_CHAPTER_HEADING_PATTERN = ^第[一二三四五六七八九十百千万零〇两\d]+(?:章|节|编|部分|篇)(?:[\s　].*)?$
_ARTICLE_HEADING_PATTERN = ^第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十\d]+)?$
_SECTION_NAME_PATTERN    = ^(?:总则|分则|附则|通则|序言|前言|目\s*录)$
```
14 条专项测试锁定（`tests/test_cleaner_headings.py`），其中含"真实导航噪音仍被剔除"的反向用例：
`首页`、`机构设置 联系我们`、`某某`、`abc`、`12`、`首页 资讯 公告 政策 法规` 仍全部删除。

---

## 四、验收 d：PDF 解析 vs 库内 HTML 解析（同一部《劳动合同法》）

对照对象：`documents#9`（approved version#164，来源 `samr.gov.cn`）× `.pdf_runs/jm_labor_contract_law.cleaned.txt`

### 最终结果（修复后）

```
① 体量对比                    PDF          HTML
  清洗后字符数                  12,196        12,149
  总块数                        327           324
  条（父块）                       98            98
  款/项（子块）                    229           226
    其中项                         80            80
    其中款                        149           146

② 条号序列对比
  ✅ 条号序列逐项一致

③ 逐条正文对比（规范化后按条号配对）
  配对条数 = 98  完全一致 = 98  有差异 = 0
  一致率 = 100.0%
```

> 修复前是 **97 条 / 96 条一致（99.0%）**，差的那 1 条就是下面差异点①。

### 差异点（≥3 个，均为实际测出）

**差异 1：MinerU 会把条文行提升为 Markdown 标题，导致条文边界漏切 1 条（已修）**

修复前 PDF 侧少 `第十七条`，因为 MinerU 输出的是：
```
## 第十七条 劳动合同应当具备以下条款:
```
而条文边界正则是 `^\s*第X条` —— 行首多了 `## ` 就匹配不上，该条被并进 `第十六条`
（PDF 第十六条 295 字 vs HTML 69 字）。修复：PDF 分支复用 `.md` 路径的
`_strip_markdown_markers`（MinerU 返回的本就是 Markdown）。修完 98 vs 98。

**差异 2：款/项的段落边界判定不同 —— PDF 侧多出 3 个子块（未修，待裁决）**

```
⑤ 逐条子块（款/项）数对比 —— 定位剩余差异
  子块数不一致的条文 = 3 条
    【第四条】PDF 子块 5 个 / HTML 子块 4 个
    【第三十条】PDF 子块 3 个 / HTML 子块 2 个
    【第八十四条】PDF 子块 4 个 / HTML 子块 3 个
```
原因（已定位到具体字符）：
```
PDF  第三十条 …国家规定，向⏎⏎劳动者及时足额支付劳动报酬。⏎⏎用人单位拖欠或者…
HTML 第三十条 …国家规定，向劳动者及时足额支付劳动报酬。⏎用人单位拖欠或者…
```
即 **PDF 版面换行处被 MinerU 还原成 `\n\n`**，而 `_split_into_paragraphs` 把 `\n\n` 当段落边界，
于是同一「款」被多切 1 段。这是**版面还原的固有噪声**，不是解析失败；
但会让"同一法条在 PDF 入库与 HTML 入库下切块数量不同"。

**差异 3：PDF 含「目录」段，库内 HTML 不含（数据差异，非缺陷）**
```
含「目录」段：PDF=True  HTML=False
```
PDF 发布件带章节目录页，网页版没有。这会让 PDF 入库多出 20 行左右的目录文本
（进正文但不含条文号，切块时落在文档级内容里）。

**差异 4（次要）：文字层直取 vs MinerU 结构化输出，字符数差 1%**
PyMuPDF 直取 12,375 字 vs MinerU 12,250 字：MinerU 会合并/丢弃页眉页脚与页码，
并对表格做结构重排，这是它的设计目标而非损失。另注：PDF 文本里的全角空格、
「目 录」这类排版空格会被保留（HTML 版没有）。

---

## 五、过程中发现并修复的 4 个真缺陷（含方案勘误）

### 缺陷 1：预签名上传 403 —— 根因是 `Content-Type`，不是 `Authorization`

**方案里写错了**（原文："★ 这一步【不能带 Authorization】，否则 403"）。实测对照：

| 形态 | 结果 |
|---|---|
| `curl -T 文件`（无 Content-Type） | **200 OK** |
| urllib 无显式头（自动补 `Content-Type: application/x-www-form-urlencoded`） | 403 `SignatureDoesNotMatch` |
| urllib 显式 `Content-Type: application/pdf` | 403 `SignatureDoesNotMatch` |
| `http.client` 不发 Content-Type | **200 OK** |

真实原因：OSS 预签名 URL 按「空 Content-Type」参与签名，urllib **只要带 data 就自动补
Content-Type**，多这一个头就签名不匹配。修法：新增 `http_retry.request_raw`，
用 `http.client` 精确控制请求头（只发调用方给的头）。同时把空 body 的 GET
改成 `data=None`，避免轮询请求被补上 `Content-Length: 0`。

### 缺陷 2：Qwen-VL 大请求体丢页（只回第一张图）

实测（同一份真实 PDF，同样 10 页）：

| 每批图片数 | 请求体 | 识别字数 | 结论 |
|---|---|---|---|
| 1 图 | ~0.4 MB | 3,816 字（前 8 页合计） | 完整 |
| 4 图 | 1.66 MB | 1,929 字 | 完整 |
| **6 图** | **2.45 MB** | **400 字** | **只剩第 1 页** |
| **8 图** | **3.28 MB** | **399 字** | **只剩第 1 页** |

**阈值在 1.7~2.4 MB 之间，且服务端会"成功返回"只含第 1 页的内容**——静默丢页。
影响：整条兜底分支从 38 条掉到 16 条（条号出现 `第五条 → 第二十八条` 的断层）。

修法：分批约束从「只看页数」改成「**页数 + 请求体积**双约束」，体积是真正生效的那个，
新增配置 `QWEN_VL_MAX_REQUEST_BYTES`（默认 1.5 MB，150 DPI 下约 3~4 页一批）。
修复后：3 批 / 4,060 字 / **38 条（完整）**。

### 缺陷 3 & 4

见第四节差异点①（Markdown 标题吃条文边界）与第三节（cleaner 吃章节短行）。

---

## 六、验收 f：全量测试与服务自检

```
########## 全量测试 ##########
595 passed, 4 warnings in 22.53s

########## 服务健康自检 ##########
✅ 容器 my-redis / milvus-standalone
✅ Redis ping  PONG
✅ MySQL 连接（127.0.0.1:3306/legal_rag）
   documents=11  document_chunks=1627  law_versions=11  document_versions=11
✅ Milvus 连接（127.0.0.1:19530）；集合 legal_documents 存在
✅ 向量数一致性  Milvus=1627 == MySQL.document_chunks=1627
✅ 版本状态统计  approved=11
=== 结果 === ✅ 全部通过（退出码 0）

########## 全仓 import 校验 ##########
扫描文件数: 94（backend/app 由 pytest 覆盖，此处跳过）
✅ 全部 app.* 导入均可解析（模块存在 + 符号存在）
```

585 → 595 的增量正好等于新增测试条数（25+26+22+14+12+6+5 = **110**），既有 485 条一条没少。

---

## 七、已知限制（如实报告，未美化）

1. **兜底分支的 `page_count` 报 0** —— **批次 23 已修**：`parse_pdf` 在 MinerU 页数为 0 时
   回退取 Qwen-VL 报告的页数（实跑兜底链路 `page_count = 10`），注释写明"页数来源随分支变化"，
   并加测试锁定；`page_count` 不在任何接口响应里（`app/api/`、`app/schemas/` 零引用），无需改接口文档。
2. **MinerU v4 不一定返回 `extract_progress`**（本次就没返回），vlm 后端结果包也不含页图；
   页数解析已按四层回退：`extract_progress.total_pages` → `layout.json.pdf_info` →
   包内 `*_origin.pdf` 页数（PyMuPDF）→ 页图数量。本次实际命中 `layout.json`。
3. **Qwen-VL 输出无结构** —— 批次 23 裁决：**不修，长期登记**（见 `data/pdf_samples/README.md`
   「已知限制」）。`qwen-vl-ocr` 只吐纯文本，"章"与"条"的层级靠文本形态维持，没有 Markdown 层级
   （兜底分支的 38 条是完全靠文本识别切出来的）；章节标题只能靠 cleaner 的结构标题白名单
   保住不被当导航行删。兜底是故障降级路径，为此在纯文本上做启发式结构补全，误判风险大于收益。
4. **PDF 尚未接入生产入库流程**：本批只把 `offline_ingest` 的 PDF 通道**接通并单测**，
   没有对真实生产 `data/labor_law_raw` 跑一次 PDF 入库（那需要 manifest 与 URL 映射）。

---

## 八、已知差异登记（本批裁决：**暂不修**）

| 差异 | 事实 | 影响面 | 结论 |
|---|---|---|---|
| **差异 2：PDF 版面换行被还原成 `\n\n`，同一「款」多切子块** | MinerU 把 PDF 的版面换行还原成 `\n\n`，而 `_split_into_paragraphs` 将 `\n\n` 当段落边界，于是**同一款被多切 1 段**。实测 3 条：第四条（5 vs 4）、第三十条（3 vs 2）、第八十四条（4 vs 3） | 只影响款级块数（3/98 条）；**条级正文 98/98 完全一致**；引用展示到**父块（条）**，用户看到的内容不变 | **暂不修**（裁决②）。理由是收益极小、启发式合并有误合风险 |

复现方式（`scripts/e2e/compare_pdf_vs_html.py` 第五节会就地打印这段）：

```
PDF  第三十条 …国家规定，向⏎⏎劳动者及时足额支付劳动报酬。⏎⏎用人单位拖欠或者…
HTML 第三十条 …国家规定，向劳动者及时足额支付劳动报酬。⏎用人单位拖欠或者…
```

另两条差异不属于缺陷，一并登记：**差异 3**（PDF 发布件带「目 录」页、库内 HTML 没有 → 属数据差异）
与**差异 4**（PyMuPDF 文字层直取 12,375 字 vs MinerU 12,250 字，差 1%，属 MinerU 合并页眉页脚/表格重排的设计行为）。

---

## 九、待裁决 → 本批裁决与执行结果

| 原待裁决项 | 你的裁决 | 执行结果 |
|---|---|---|
| ① `requirements.txt` 与本地环境不一致（13 项只 4 项对上） | **以本地为准反向更新** | ✅ 已按本地 `rag` 环境实测重写，14 条全 `==`；dry-run 解析全部 "already satisfied"；重跑 **598 passed** + check_services 退出码 0。详见第十一节 |
| ② 差异 2 是否修 | **不修，但要登记** | ✅ 已登记（见第八节） |
| ③ 临时脚本 / `.pdf_runs/` / `.pdf_probe/` 处置 | **升格 + 清理** | ✅ 4 个脚本升格进 `scripts/eval/`、`scripts/e2e/`；`.pdf_runs/` 已清；4 份 PDF 早已在 `data/pdf_samples/`（含 README） |
| ④ 兜底分支 `page_count` 报 0 | 未单独立项（本轮补充意见同批下达） | ✅ 已修 + 3 条新测试锁定；真实兜底分支复跑实测 `page_count = 10` |
| ⑤ multi_turn 纳入评测（3~5 条，先交审） | **新增题目先交我审** | ⏳ 题目草案已出：`reports/batch22_multi_turn_candidates.md`（4 条 + 1 条备用，golden 已逐条核对在库）——**等你审** |

**仍待你定（本轮新发现，未擅自处理）**

1. **项目根的 `probe_fetch_pdf.py`**（78 行）：批次 22 找 PDF 时的一次性抓取探针，作用是
   "按公开 URL 重新下载这 4 份样本并校验页数/文字层"。不在你给的清理清单里，故**保留未动**——
   要不要一起升格成 `scripts/e2e/fetch_pdf_samples.py`（它是 `data/pdf_samples/` 的复现入口），还是直接清掉？
2. **`requests` 未声明**：`scripts/e2e/` 下 3 个脚本 `import requests`，但 `requirements.txt` 只服务
   `backend/`，此前也没列它（本地有 2.34.2）。要不要补进 requirements（或另建 `requirements-scripts.txt`）？
3. **兜底分支输出无结构**：`qwen-vl-ocr` 只吐纯文本，没有 Markdown 层级，"章/条"层级全靠文本形态维持
   （本批 38 条切块是靠文本识别切出来的）。要不要后续给兜底分支加"结构补全"？

---

## 十、复现命令

```bash
# 0) 依赖（批次 22：PDF 解析链路）
pip install -r backend/requirements.txt

# 1) 主路径（MinerU，真实远程调用）
python scripts/e2e/pdf_parse_main_path.py --pdf jm_labor_contract_law.pdf

# 2) 兜底分支（假 MinerU + 真实 Qwen-VL OCR）
python scripts/e2e/pdf_parse_qwen_fallback.py --pdf jm_lc_implement_regulation.pdf

# 3) PDF vs 库内 HTML 对照（默认再跑一次 MinerU；--cleaned 可复用产物、不联网）
python scripts/e2e/compare_pdf_vs_html.py

# 4) 多轮 + 摘要 + 改写 链路检查（真实检索 + 真实 LLM）
python scripts/eval/multi_turn_summary_check.py --rounds 14

# 5) 回归
cd backend && PYTHONPATH= python -m pytest tests -q --no-header -p no:cacheprovider \
  --basetemp="C:/Users/92842/AppData/Local/Temp/pytest_b22"
python scripts/check_services.py
python scripts/check_imports.py
```
```

---

## 十一、本批收尾执行明细

### 1) `requirements.txt` 按本地实测重写（裁决①）

采集方式：`C:/Users/92842/anaconda3/envs/rag/python.exe -m pip freeze`（2026-09-21）。

| 包 | 改前（requirements） | 改后（= 本地实测） |
|---|---|---|
| fastapi | 0.138.1 | **0.104.1** |
| starlette | 1.3.1 | **0.27.0** |
| pydantic | 2.13.4 | **2.5.0** |
| uvicorn[standard] | 0.49.0 | **0.24.0** |
| SQLAlchemy | 2.0.34 | **2.0.51** |
| PyMySQL | 1.1.1 | **1.2.3** |
| pymilvus | 2.5.18 | **2.6.17** |
| pytest | 8.4.2 | **7.4.3** |
| httpx | 0.28.1 | **0.25.2** |
| redis / jieba / rank-bm25 / PyMuPDF | 5.3.1 / 0.42.1 / 0.2.2 / 1.28.0 | 不变（本来就一致） |

- ⚠️ **你的示例值与实测不符**：你写的是"fastapi 0.133.1、starlette 1.0.1"，但我在本机
  **三个解释器里都测过**：`rag` env = **fastapi 0.104.1 / starlette 0.27.0**（就是跑 598 测试那个），
  `anaconda3 base` = 0.138.1 / 1.3.1（= 改前 requirements 的来源），另两个解释器没装 fastapi。
  **没有一处是 0.133.1 / 1.0.1**。按你的口径"以实测为准"，我锁的是 `rag` env 的值 —— 若你手上那份
  0.133.1/1.0.1 来自容器镜像或另一台机器，请告诉我，我再对齐（这属于"锁给谁用"的口径问题）。
- 依赖自洽性：`pip install --dry-run -r backend/requirements.txt` → 全部 "Requirement already satisfied"，
  退出码 0（说明这 14 条与本地环境完全一致、彼此无冲突）。
- 回归：**598 passed, 4 warnings in 14.47s**；`check_services.py` 退出码 0
  （MySQL 1627 条 = Milvus 1627、approved=11）；`check_imports.py` 94 文件全绿。
- `.env.example` 顶部注释**没有**提到任何版本号（只有"版本库"= 代码仓库那处措辞），故无需改动。
- 副作用如实说明：容器将以 fastapi 0.104.1 / starlette 0.27.0 构建（比原来旧）。
  若要整体升级，建议另开一批做"升级 + 全量回归 + 容器构建验证"，本批只做"两端对齐"。

### 2) 四个临时脚本升格 + 清理（裁决③）

| 原（项目根，临时） | 新（正式，带退出码） | 说明 |
|---|---|---|
| `b22_summary_multiturn.py` | `scripts/eval/multi_turn_summary_check.py` | 轮数可配（`--rounds`，默认 14）；新增**每轮查询改写记录**；区分 0/1/2/3 四种退出码 |
| `b22_pdf_main_path.py` | `scripts/e2e/pdf_parse_main_path.py` | 加 `--pdf/--title/--out-dir`；产物默认写系统临时目录 |
| `b22_pdf_qwen_fallback.py` | `scripts/e2e/pdf_parse_qwen_fallback.py` | 同上；分支不符预期时以退出码 1 失败 |
| `b22_compare_pdf_html.py` | `scripts/e2e/compare_pdf_vs_html.py` | **不再依赖 `.pdf_runs/`**：默认自己现跑一次 MinerU，或 `--cleaned 文件` 复用产物（不联网） |

清理（全部走 Windows 回收站 API，未硬删）：项目根 4 个 `b22_*.py`、`.pdf_runs/`、
被替代的 `reports/b22_summary_multiturn_trace.txt`，以及本次运行生成的
`scripts/e2e/__pycache__`、`scripts/eval/__pycache__`。

⚠️ **一处没能自证的地方**：`SHFileOperationW` 调用返回码是 **2**（非 0），但 6 个目标清理后
均已确认不存在；沙箱不允许读取 `C:\$Recycle.Bin`（PowerShell COM 被安全策略拦截、直接遍历无输出），
所以"能否从回收站还原"我**无法自证**。后续清理我会改用"先复制到 `.dev_bakNN/` 再删"的方式，可验证性更好。

四个新脚本**都真实跑过**（不是纸面迁移）：

| 脚本 | 实跑结果 | 退出码 |
|---|---|---|
| `pdf_parse_main_path.py` | 98 条 / 229 子块（80 项）/ 章节标题 10 项全 ✔ / 非空行 262→262 删 0 行 | 0 |
| `pdf_parse_qwen_fallback.py` | `branch=qwen-vl`、10 页、3 批、38 条 / 63 子块（19 项）、**page_count=10** | 0 |
| `compare_pdf_vs_html.py` | 条号序列逐项一致、**98/98 条 100% 一致**、3 条子块差异（已登记） | 0 |
| `multi_turn_summary_check.py` | 摘要第 10 轮产出、前情第 11 轮注入（提示词 1239 字）、**第 11 轮改写生效** | 0 |

新脚本清单与"作用 / 前置 / 可重复性 / 跑完怎么清理"已写进 `scripts/e2e/README.md`
与 `scripts/eval/README.md`（新建）；`data/pdf_samples/README.md` 的复现命令同步改为新路径。

### 3) 兜底分支 `page_count` 修正（补充意见④）

- 改动：`app/ingest/pdf_parser.py` —— `parse_pdf` 在 MinerU 页数为 0 时改用 Qwen-VL
  `last_trace["page_count"]`（新增 `_qwen_vl_page_count` 辅助函数，读不到就保持 0，不抛异常），
  注释写明"页数来源随分支变化"。
- **先确认过接口面**：`page_count` 只出现在 `app/ingest/*`（`ParsedDocument` 字段）与
  `app/models/*`（客户端痕迹），`app/api/` 与 `app/schemas/` **零引用**，即**不进任何接口响应**，
  因此不涉及接口文档变更。
- 新增 3 条测试：MinerU 报 0 时用 Qwen-VL 的页数 / Qwen-VL 也没报时退回 0 / 痕迹不是字典时不崩。
- 真实链路复验：兜底脚本实测 `parse_document → page_count = 10`（修复前该值是 **0**）。

### 4) 多轮 + 摘要 + 改写（补充意见⑤）

- 已有产出：`scripts/eval/multi_turn_summary_check.py` 实跑 14 轮，输出存档
  `reports/batch22_multi_turn_check_trace.txt`。
- 顺带修掉一个**探针自身的假阴性**：检索侧就拒答的轮次（不调 LLM）原本会被判成"没注入前情"，
  现记为 `前情=n/a` 并单独计入退出码 2 的判定，避免把"没测到"当"没生效"。
- 评测集题目**只出草案未落地**：`reports/batch22_multi_turn_candidates.md`
  （4 条候选 + 1 条备用；golden 已逐条 SQL 核对存在于 approved 版本；含题面 schema、
  `run_eval` 最小改动清单、指标口径、以及"反向对照"验收办法）。**等你审后再写入评测集。**

---

## 十二、附录：第一组验收（批次 21 收尾三件）

### 1) 删除存储类 5 键 ✅

删除对象：`FILE_STORAGE_ROOT` / `DOCUMENT_STORAGE_PATH` / `PARSED_STORAGE_PATH` /
`BACKUP_STORAGE_PATH` / `MAX_UPLOAD_SIZE_BYTES`（`.env` 与 `backend/.env.example` 两边都清）。

```
$ grep -rn "FILE_STORAGE_ROOT\|DOCUMENT_STORAGE_PATH\|PARSED_STORAGE_PATH\|BACKUP_STORAGE_PATH\|MAX_UPLOAD_SIZE_BYTES" \
    .env backend/.env.example backend/app
✔ 5 键无任何残留
```
- 删除前已确认它们**零读取**（批次 21 报告已记录：生效但无代码引用），因此删除不影响任何链路。
- 全量测试与 `check_services.py` 在删除后仍全绿（见 4）。
- ⚠️ 若生产 `.env` 里原先手工设过这几键，部署时同步删除即可（代码侧已无读取点，留着也无害）。

### 2) 清理本批临时脚本 ✅

```
$ ls b21_*.py .dev_bak21/
(.dev_bak21 不存在)     # b21_*.py 4 个：0 命中
```
`b21_*.py` × 4 与 `.dev_bak21/` 均已清理；项目根不再有 b21 残留。

### 3) 摘要开/关 20 条质量对拍 ✅

同一 20 条评测集、同一检索口径（向量 20 + 关键词 20 → RRF → 重排，指标基于 top10），
仅切换 `SESSION_SUMMARY_ENABLED`：

| 指标 | 摘要关（基线） | 摘要开 | 变化 |
|---|---|---|---|
| Recall@5 | 1.0000 | 1.0000 | — |
| MRR@10 | 0.7825 | 0.7825 | — |
| **引用正确率** | **1.0000**（越界 0 / 引用总数 100） | **1.0000**（越界 0 / 引用总数 96） | — |
| 误拒率 | 0.2000（4 条无引用） | 0.1000（2 条无引用） | —（波动，非摘要作用） |
| 无引用回答数 | 4 | 2 | —（波动，非摘要作用） |
| 时效题越界引用 | 0 | 0 | — |

- **检索侧零影响**：Recall/MRR 逐位相同，说明摘要只作用于提示词构造之后，未触碰检索链路。
- ⚠️ **这组对拍证明不了"摘要的影响"**：`evaluation/run_eval.py` 每题用独立 session（`eval_<题号>`）
  且直接调 `ChatService.chat`，**不走 `api/chat_persistence.persist_turn`** —— Redis 里探针 key 数为 0，
  即"开启"那一轮的摘要**从未参与任何一次推理**。两轮差异（无引用 4→2、引用总数 100→96）
  只能归为 LLM 采样波动，**不能拿来当"摘要提升了回答质量"的证据**。
- 因此另建多轮探针补真实证据（见下一条）。
- 报告：`reports/latest_eval_b22_summary_on.md` / `reports/latest_eval_b22_summary_off.md`。

**补充证据：真实多轮链路（`scripts/eval/multi_turn_summary_check.py --rounds 14`，本次实跑，原始输出已落盘）**

14 轮连续真实问答（真实检索 + 真实 LLM），统计窗口（`ltrim -20`）满后真实触发摘要：

```
=== 环境 ===
SESSION_SUMMARY_ENABLED(实测) = True
短期记忆窗口 = 20 条消息；摘要触发实测最早第 10 轮、前情注入最早第 11 轮
探针 user_id=probe_b22_user  session_id=probe_b22_multiturn  轮数=14
（前情=n/a 表示本轮在检索侧就拒答、未调用 LLM，提示词无从观测）
[ 1] 消息= 2 摘要=  0字 新摘要=无 前情=no 改写=no 引用=5
[ 2] 消息= 4 摘要=  0字 新摘要=无 前情=no 改写=no 引用=5
[ 3] 消息= 6 摘要=  0字 新摘要=无 前情=no 改写=no 引用=5
[ 4] 消息= 8 摘要=  0字 新摘要=无 前情=n/a 改写=no 引用=0
[ 5] 消息=10 摘要=  0字 新摘要=无 前情=no 改写=no 引用=5
[ 6] 消息=12 摘要=  0字 新摘要=无 前情=no 改写=no 引用=5
[ 7] 消息=14 摘要=  0字 新摘要=无 前情=no 改写=no 引用=5
[ 8] 消息=16 摘要=  0字 新摘要=无 前情=no 改写=no 引用=5
[ 9] 消息=18 摘要=  0字 新摘要=无 前情=no 改写=no 引用=5
[10] 消息=20 摘要= 69字 新摘要=有 前情=no 改写=no 引用=5
[11] 消息=20 摘要= 69字 新摘要=无 前情=YES 改写=YES 引用=5
      改写：那这个要赔多少？ → 孕妇被辞退有什么特别保护 这个要赔多少 怎么算
[12] 消息=20 摘要= 69字 新摘要=无 前情=n/a 改写=no 引用=0
[13] 消息=20 摘要=300字 新摘要=有 前情=YES 改写=no 引用=5
[14] 消息=20 摘要=300字 新摘要=无 前情=YES 改写=no 引用=5
```

① LLM 真实压出来的摘要原文（存于 Redis，**非预置**；该轮 `349 -> 300` 字被上限截断）：

```
一名劳动者咨询中国大陆劳动法，涉及经济补偿、工龄、月工资超社平三倍、免付经济补偿、违法解除赔偿、
试用期期限与工资、加班费基数、未休年假折算、孕妇辞退保护。助手已答：经济补偿按工龄；免付经济补偿
因未找到直接法条无法确认；违法解除赔偿金为经济补偿标准二倍，不重复支付，年限自用工之日起，可要求
继续履行；试用期工资可低于正式工资，但不得低于同岗位最低档或合同工资80%且不低于当地最低工资……
```

② 某一轮实际注入提示词的「# 本次会话前情」段全文（**该轮提示词总长 1239 字**）：

```
# 本次会话前情（早前轮次的压缩摘要，仅供理解上下文，不得替代法源清单）
（同上摘要原文，位于「# 法源清单（本次检索结果）」段之前）
```

- 两处边界现象（都是**按设计降级**，未影响链路）：
  ① 窗口满后**首次**摘要生成尝试因 LLM 返回空回答而失败
  （`LlmApiError: 大模型返回了空回答（可能是推理过程占满了 max_tokens 额度）`）
  → 按设计保留旧摘要、不推进 `summarized_turns`；下一次触发轮重试成功；
  ② 摘要超 300 字被截断（上限生效）。
- **改写记录**：1/14 轮发生改写（默认题目集只有末两条是省略式追问；全是自足问题的题目集
  跑不出改写 —— 这也是 `reports/batch22_multi_turn_candidates.md` 里题面设计要带指代的依据）。
- 原始输出：`reports/batch22_multi_turn_check_trace.txt`。

### 4) 第一组收尾回归 ✅

```
第二组收尾时：595 passed, 4 warnings in 22.53s
本批（裁决④）收尾后复跑：598 passed, 4 warnings in 14.47s
=== 服务自检结果 === ✅ 全部通过（退出码 0）
✅ 全仓 import 校验：94 文件全部可解析
```

### 第一组可复现命令

```bash
# 5 键残留检查（期望 0 命中）
grep -rn "FILE_STORAGE_ROOT\|DOCUMENT_STORAGE_PATH\|PARSED_STORAGE_PATH\|BACKUP_STORAGE_PATH\|MAX_UPLOAD_SIZE_BYTES" \
  .env backend/.env.example backend/app

# 摘要开/关对拍（20 条；注意：run_eval 每题独立 session，摘要不会参与，见上文说明）
python evaluation/run_eval.py --limit 20 --tag b22_summary_on     # .env: SESSION_SUMMARY_ENABLED=true
python evaluation/run_eval.py --limit 20 --tag b22_summary_off    # .env: SESSION_SUMMARY_ENABLED=false

# 摘要真实多轮证据（14 轮真实问答，输出已存 reports/batch22_multi_turn_check_trace.txt）
python scripts/eval/multi_turn_summary_check.py --rounds 14

cd backend && PYTHONPATH= python -m pytest tests -q --no-header -p no:cacheprovider \
  --basetemp="C:/Users/92842/AppData/Local/Temp/pytest_b22"
python scripts/check_services.py
python scripts/check_imports.py
```

样本与来源见 `data/pdf_samples/README.md`；中间产物落 `.pdf_runs/`。
