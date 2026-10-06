# PDF 回归样本（data/pdf_samples/）

批次 22「用真实 PDF 验证 PDF 解析链路」的回归素材。四份**全部来自公开政府网站**，
可溯源、可公开访问；都是真实发布件，没有手造或项目内文件。

用途：
- `MinerU 主路径`（有文字层 → 结构化解析）回归；
- `Qwen-VL 兜底路径`（渲染页图 → OCR）回归；
- `PDF 解析 vs 库内 HTML 解析`对照（第 1~3 份在法规库里有对应文档）。

## 样本清单

| 文件 | 法规名 | 页数 | 大小 | 文字层 | 库内对应文档 |
|---|---|---|---|---|---|
| `jm_labor_contract_law.pdf` | 中华人民共和国劳动合同法 | 27 | 282,029 B | 有（12,375 字） | `documents#9`（HTML） |
| `jm_lc_implement_regulation.pdf` | 中华人民共和国劳动合同法实施条例 | 10 | 261,817 B | 有（4,617 字） | `documents#3`（HTML） |
| `jm_labor_law.pdf` | 中华人民共和国劳动法 | 21 | 273,566 B | 有（8,952 字） | `documents#10`（HTML） |
| `nc_labor_contract_general.pdf` | 劳动合同（通用）示范文本（人社部编制版） | 7 | 150,930 B | 有（3,591 字） | 无 |

「文字层」一列是用 PyMuPDF **逐页 `page.get_text()` 字符数直接相加**（不 strip）得到的，
> 0 即说明不是纯扫描件，这也是判断该走 MinerU 还是 Qwen-VL 的依据。
口径务必别改成"逐页 strip 后相加"：那样每页会少 1 个换行，27 页的样本会差 27 字，
容易被误读成"样本变了"。`scripts/e2e/fetch_pdf_samples.py` 按本口径校验（2% 容差），
预期值与该脚本的 `TARGETS` 常量两处必须一起改。

## 来源（逐份可溯源）

- **第 1~3 份**：江门市人力资源和社会保障局「劳动法律和政策文件」栏目
  页面：<https://www.jiangmen.gov.cn/bmpd/jmsrlzyhshbzj/zcfg/ldgx/content/post_2982353.html>
  该页面以附件形式提供三部法律的 PDF：
  - 附件 `287575` → `jm_labor_contract_law.pdf`
  - 附件 `287576` → `jm_lc_implement_regulation.pdf`
  - 附件 `287574` → `jm_labor_law.pdf`
- **第 4 份**：南昌市人民政府 2025-07 公布的人社部编制版示范文本
  页面：<https://www.nc.gov.cn/ncszf/shbz/202507/65ce5e3bd5f546fe8f446c374221f6d7.shtml>

## 复现命令

```bash
# 0) 重新拉样本（从下方"来源"里的原始 URL 下载，并校验页数/文字层与本 README 一致）
python scripts/e2e/fetch_pdf_samples.py            # 缺哪份下哪份；--force 全部重下；--verify-only 只校验不联网

# MinerU 主路径（真实远程调用）
python scripts/e2e/pdf_parse_main_path.py --pdf jm_labor_contract_law.pdf

# Qwen-VL 兜底路径（假 MinerU 触发 + 真实 Qwen-VL OCR）
python scripts/e2e/pdf_parse_qwen_fallback.py --pdf jm_lc_implement_regulation.pdf

# PDF vs 库内 HTML 对照（默认再跑一次 MinerU；加 --cleaned 可复用产物、不联网）
python scripts/e2e/compare_pdf_vs_html.py
```

前置：`.env` 里 `MINERU_API_KEY` / `QWEN_VL_API_KEY` 有效；
`pip install PyMuPDF==1.28.0`（页图渲染 + 文字层对照都靠它）。
四个脚本都不写库、不写 Redis；解析类中间产物写在 `--out-dir`（默认系统临时目录，脚本会打印路径），
跑完整目录删除即可。脚本细节见 `scripts/e2e/README.md`。

## 已知限制

- **Qwen-VL 兜底分支的输出没有 Markdown 层级**（批次 23 登记，暂不修）：
  MinerU 返回 `full.md`，章节/条文层级完整，因此 `_parse_pdf_file` 能靠 Markdown 标记
  保住条文边界；而 Qwen-VL 返回的是**纯 OCR 文本**，没有 `#` 层级，
  兜底分支的 38 条 / 19 项完全靠"第X条"文本形态切出来。
  影响：兜底分支的章节标题（"第一章""总则"）无法作为结构被识别，只能靠 cleaner 的
  结构标题白名单保住不被当导航行删掉；若后续 OCR 输出格式变化，条文边界识别会变脆。
  不修的原因：兜底分支是**故障降级路径**（MinerU 挂了才走），
  为此在纯文本上做启发式结构补全，误判风险大于收益；改由本清单长期登记。

## 找 PDF 的渠道备忘（踩过的坑）

- Bing / 百度 / 360 / 搜狗的结果页都是 JS 渲染或降级，`curl` 直接抓拿不到真实附件链接；
- 有效路径是：搜索引擎（WebSearch）找到**政府页面** → `curl` 抓该页面 HTML →
  抠 `<a ... .pdf>` 锚文本，才能把「法规名 ↔ 附件 ID」对上；
- 司法部详情页、政府网政策页、最高法公报页多数**不带 PDF 附件**；
  地方人社局/地方政府网站的「示范文本 / 政策文件」栏目命中率最高。
