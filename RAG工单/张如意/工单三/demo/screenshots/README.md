# 演示截图目录

> 工单编号：**人工智能NLP-RAG-PDF文档的表格解析及检索优化**

本目录用于存放演示视频与验收报告中的截图。按下面的清单拍摄并命名，
录制 `demo/演示脚本.md` 时可直接对应引用。

---

## 建议截图清单

| 文件名 | 对应分镜 | 画面内容 | 拍摄方式 |
| --- | --- | --- | --- |
| `01-工单目标与四个问题.png` | 01 | 4 个表格类问题清单 | 文档或终端 |
| `02-文本层丢失结构.png` | 02 | p22 纯文本层输出（序号/项目名/金额各自成行） | 终端打印 |
| `03-表格抽取终端输出.png` | 03 | `python table_extractor.py` 的运行日志 | 终端 |
| `04-表格Markdown文件.png` | 03 | `results/tables/` 目录 + 打开一张 .md | 资源管理器 + 编辑器 |
| `05-表头列错位修复前后.png` | 03 | 修复前 6×9 / 修复后 6×3 对照 | 文档或终端 |
| `06-跨页合并事件.png` | 04 | `table_inventory.json` 的 `merge_events`（merged 与 rejected 各一条） | 编辑器 |
| `07-索引统计.png` | 05 | `index_stats.json` 的 chunk_types 与 table chunk 明细 | 编辑器 |
| `08-table_qa报告.png` | 06 | `results/table_qa.md` 中 id 1 的完整条目 | 浏览器/预览 |
| `09-检索精确度.png` | 06 | 精确度四项指标的特写（可裁剪） | 报告截图裁剪 |
| `10-附表的表格原文渲染.png` | 06 | 表格 Markdown 原样展示 | 编辑器预览 |
| `11-消融实验对比表.png` | 07 | `table_ablation.md` 第二节准确率对比表 | 浏览器/预览 |
| `12-逐题明细.png` | 07 | 消融报告第四节逐题明细 | 浏览器/预览 |
| `13-Web问答界面.png` | 08 | `serve.py` 首页 + 一次完整提问结果 | 浏览器全屏 |
| `14-表格渲染成HTML.png` | 08 | 检索片段里表格渲染效果特写 | 浏览器 |
| `15-英文提问.png` | 09 | 英文问题 + 英文回答 | 浏览器 |
| `16-离线抽取式兜底.png` | 09 | 无 API Key 时运行 `table_qa.py --no-llm` | 终端 |
| `17-验收自检清单.png` | 10 | `docs/验收对照表.md` 的清单逐条打勾 | 编辑器 |

---

## 命名与规格要求

- 命名：`序号-内容简述.png`，序号与上表一致，便于与分镜对齐；
- 分辨率：不低于 1920×1080；界面截图避免缩放到模糊；
- 终端截图：把字号调到 ≥ 16pt，避免小字无法辨认；
- 敏感信息：若环境中配置了真实 API Key，截图前请确认终端中
  不出现 `sk-` 开头的密钥字符串。

---

## 快速生成界面截图

```bash
cd "工单作业/工单03-表格解析及检索优化/src"
python serve.py --port 8013
# 浏览器打开 http://127.0.0.1:8013
# 点击右侧演示问题，等待答案与表格渲染完成后截图
```

若需要无人值守批量截图，可用 Playwright：

```bash
pip install playwright -i https://pypi.tuna.tsinghua.edu.cn/simple
playwright install chromium
```

```python
# 示例：对 4 个表格类问题逐个截图
from playwright.sync_api import sync_playwright

QUESTIONS = [
    "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
    "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
    "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？",
    "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
]

with sync_playwright() as pw:
    page = pw.chromium.launch().new_page(viewport={"width": 1920, "height": 1080})
    page.goto("http://127.0.0.1:8013")
    for i, q in enumerate(QUESTIONS, 1):
        page.fill("#q", q)
        page.click("#send")
        page.wait_for_selector(".card .answer", timeout=60000)
        page.wait_for_timeout(1200)          # 等表格渲染完
        page.screenshot(path=f"../demo/screenshots/qa-{i}.png", full_page=True)
```
