# -*- coding: utf-8 -*-
"""把附录 A 的转录从卡片里导出成可读文档 -> data/processed/vision_transcribe_appendixA.md

为什么要有这份文件：
  附录 A 的字段名此前只存在于 appendixA_cards.jsonl 的 record_form 里，没有可核对的文本。
  附录 B 的教训是——我单通道转录在附录 A 上错了 4 处（播种量/分组名等），
  留一份人能直接对着原图核的文档，比埋在 JSON 里稳妥。

数据源：data/processed/appendixA_cards.jsonl（已含 MinerU 交叉验证后的修正）
不改任何值，只做导出与排版。

【模块说明（补充）】
  输入：data/processed/appendixA_cards.jsonl（每行一张附录 A 知识卡，含
        record_form 表单结构与 source.page/channel 转录信息）
  输出：data/processed/vision_transcribe_appendixA.md（人可读的对照文档：
        逐作物逐表的字段转录 + 跨标准差异表 + 交叉验证结论）
  本脚本无函数，自上而下顺序执行；`L`/`w` 是「文档行缓冲」写法——
  所有输出先 append 进列表 L，最后一次性 join 写盘。
"""
import json
import os
from collections import defaultdict

P = "data/processed"                                     # 处理产物根目录
OUT = os.path.join(P, "vision_transcribe_appendixA.md")  # 输出文档路径
CROPS = ("黄瓜", "辣椒", "大蒜")                          # 三份国标对应的作物，导出顺序

cards = defaultdict(list)                                # {作物: [卡片, ...]}
for line in open(os.path.join(P, "appendixA_cards.jsonl"), encoding="utf-8"):
    c = json.loads(line)
    cards[c["crop"]].append(c)                           # 逐行读 JSONL，按作物归组
for v in cards.values():
    v.sort(key=lambda c: int(c["record_form"]["table_no"].split(".")[1]))
    # 表号形如 "A.12"，split(".")[1] 取点后的数字并转 int —— 按表号数值升序排，避免 "A.10" 排在 "A.2" 前

L = []                                                   # 文档行缓冲：先攒行，最后统一写盘
w = L.append                                             # w(...) = 往缓冲区追加一行，纯属排版简写
w("# 附录 A 生产记录表格转录（三份标准）\n")
w("> **用途**：附录 A 表头字段的可核对文本。附录 B 的对应文件是 `vision_transcribe_appendixB.md`。")
w("> **数据源**：`data/processed/appendixA_cards.jsonl` 的 `record_form` 字段——本文档是导出，不是重新转录。")
w("> **核对状态**：已与 MinerU 3.4.5 全量输出交叉比对，发现并修正 4 处（见第四节）。")
w("> **图片留档**：`logs/_verify_png/appA_*.png`")
w("")
w("## 转录方式\n")
w("| 通道 | 含义 | 页 |")
w("| --- | --- | --- |")
# 下两行是「转录通道」说明表：text=文本层直抽；vision=字形码/旋转页只能看图逐格转录
w("| `text` | 文本层能抽出字，pymupdf 直抽 | 三份的 p8–p10、p13、p15–p17，以及黄瓜 p14/p16/p18 |")
w("| `vision` | **字形码页 / 旋转 90° 横向排版**，文本层抽不出字，靠渲染原图逐格转录 | p11、p12、p14(辣椒·大蒜)、p18(辣椒·大蒜)、p19(黄瓜) |")
w("")
w("⚠️ `vision` 那 11 页是本次的**唯一来源**——没有第二通道可交叉（MinerU 同样读图，两者会犯同类错误）。")
w("")
w("---\n")

for crop in CROPS:                                       # 逐作物导出该作物的全部附录 A 表
    w(f"\n## {crop}（{cards[crop][0]['std_no']}）\n")    # 取该作物第一张卡里的标准号作标题
    for c in cards[crop]:
        rf = c["record_form"]
        w(f"### {rf['table_no']}　{rf['table_name']}\n") # 表号 + 表名作小节标题
        w(f"- **PDF 页**：p{c['source']['page']}　**通道**：`{c['source']['channel']}`")
        if rf["header_items"]:                           # 有分组表头/抬头才输出该行
            w(f"- **抬头／分组表头**：{'、'.join(rf['header_items'])}")
        w(f"- **列**（{len(rf['columns'])} 项）：")
        for i, col in enumerate(rf["columns"], 1):       # 列名逐条编号列出
            w(f"  {i}. {col}")
        if rf["footnote"]:                               # 有表注才输出
            w(f"- **表注**：{rf['footnote']}")
        w("")

w("\n---\n")
w("\n## 三、跨标准差异（**表号相同 ≠ 表相同**）\n")
w("三份标准的附录 A 表号→表名映射**并不一致**，索引/切分时不得按表号对齐：\n")
w("| 表名 | 黄瓜 | 辣椒 | 大蒜 |")
w("| --- | --- | --- | --- |")
# 每个元组是「同一张表在三份标准里的可能叫法」——用 endswith/包含做多别名匹配
names = [("生产基地田间农事活动记录表",), ("产品采收及流向记录表",),
         ("农药残留等有害物质检测结果记录表", "产品农药残留等有害物质检测结果表"),
         ("剩余药液或清洗废液处理记录表", "剩余农药或清洗废液处理记录表")]
for tup in names:
    row = [tup[0]]                                       # 表名占第一列
    for crop in CROPS:
        # 在该作物的卡片里找表名匹配（完全以别名结尾 或 别名是表名子串）的表号
        hit = [c["record_form"]["table_no"] for c in cards[crop]
               if any(c["record_form"]["table_name"].endswith(t) or t in c["record_form"]["table_name"]
                      for t in tup)]
        row.append(hit[0] if hit else "—")               # 找到取第一个表号，找不到画破折号
    w("| " + " | ".join(row) + " |")                     # 拼成一行 Markdown 表格

w("""
另有两处**表结构本身不同**，已分别建卡，未做统一：

1. **表A.1 基本情况记载表**：黄瓜是「报告编号 + 评定」两列；辣椒/大蒜是「报告编号 + 报告日期 + 评定结论」三列
2. **田间农事活动记录表**：黄瓜（A.12）分组表头为 `田间农事活动时使用农药化肥`、子列为 `农药(肥料)名称`；
   辣椒/大蒜（A.7）分组表头为 `田间农事活动记录`、子列为 `肥料名称`
3. **有害生物防治记录表**：黄瓜（A.11）是「田间调查 + 化学防治措施」双段结构；
   辣椒/大蒜（A.11）是单表 + `防治措施` 分组表头
""")

w("\n---\n")
w("\n## 四、交叉验证结论（MinerU 比对后修正）\n")
w("附录 A 曾**只有我单通道转录**，未经检验。全量 MinerU 跑完后逐项比对，**我错 4 处**：\n")
w("| 位置 | 我原先 | 原文实际 | 性质 |")
w("| --- | --- | --- | --- |")
# 下表是 4 处已修正错误的台账（形近词/漏抽/混同等），供回溯核对
w("| 三份的田间农事活动表 | `种植量 kg/667m²` | **`播种量 kg/667 m²`** | 形近词 |")
w("| 黄瓜 A.12 | 分组名 `田间农事活动记录` | **`田间农事活动时使用农药化肥`** | 我把三份当成了同一张表 |")
w("| 辣椒/大蒜 A.11 | 无分组表头 | **`防治措施`** | 漏抽 |")
w("| 辣椒/大蒜 A.7 | 列名 `农药(肥料)名称` | **`肥料名称`** | 与黄瓜 A.12 混同 |")
w("")
w("**根因**：我当初看 `appA_田间农事活动记录.png` 那张三份并排的对照图时，")
w("看到三张表长得像，就判定「结构一致」，只核了表名没核分组表头与列名。")
w("实际上三份同名不同构——这与 A.1 的情况一样，是这批国标的普遍特征。\n")

txt = "\n".join(L) + "\n"                                # 行缓冲拼接成完整文档（末尾补换行）
open(OUT, "w", encoding="utf-8").write(txt)
print(f"已写入 {OUT}（{len(txt)} 字符）")
for crop in CROPS:
    v = sum(1 for c in cards[crop] if c["source"]["channel"] == "vision")  # 统计该作物走视觉通道的表数
    print(f"  {crop}: {len(cards[crop])} 张表，其中 vision {v}")
