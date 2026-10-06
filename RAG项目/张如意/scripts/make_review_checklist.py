# -*- coding: utf-8 -*-
"""生成人工抽检清单 -> data/processed/人工抽检清单.md

三块：
  1. 附录B 25 张卡的数值列（安全间隔期按药剂名连接，需逐格核对）
  2. 正文 28 张高危卡的归类
  3. 附录A 11 张视觉转录卡的字段名

【模块说明（补充）】
  输入（均在 data/processed/ 下，逐行 JSONL）：
    - appendixB_cards.jsonl  附录 B（农药防治表）知识卡
    - body_cards.jsonl       正文条款知识卡
    - appendixA_cards.jsonl  附录 A（记录表）知识卡
  输出：data/processed/人工抽检清单.md —— 供人工对照 logs/_verify_png/ 的原页图
        逐格核对转录数值与分类，核对结果写在表格预留的空列里。
  本脚本无函数，自上而下顺序执行；`L`/`w` 为文档行缓冲写法，最后统一写盘。
"""
import json
import os

P = "data/processed"
# 一次性读入三份卡片库：B=附录B、O=正文（Original/正文条款）、A=附录A
B = [json.loads(l) for l in open(os.path.join(P, "appendixB_cards.jsonl"), encoding="utf-8")]
O = [json.loads(l) for l in open(os.path.join(P, "body_cards.jsonl"), encoding="utf-8")]
A = [json.loads(l) for l in open(os.path.join(P, "appendixA_cards.jsonl"), encoding="utf-8")]

L = []                # 文档行缓冲
w = L.append          # w(...) = 追加一行，排版简写
w("# 人工抽检清单（高危数值列 ≥20%）\n")
w("> 核对方法：对照 `logs/_verify_png/` 的原页图逐格比对。")
w("> 附录B 图：`cucumber_p20` / `cucumber_p21` / `chili_p20` / `chili_p21` / `garlic_p20_cw`；")
w("> 附录A 图：`appA_A4_人员登记表` / `appA_A5_投入品出入库` / `appA_田间农事活动记录` / `appA_A11_有害生物防治记录表`。")
w("> 「核对」列留空，无误写 ✓，有误直接写下正确值。\n")

# ------------------------------------------------ 1. 附录 B
w("---\n\n## 一、附录 B 共 25 张 —— 数值列逐条核对\n")
w("重点看：**剂量/浓度**、**稀释**、**安全间隔期**、**每茬最多次数**、**施药间隔**。\n")
w("> ⚠️ 「安全间隔期」在原表里是按**药剂名**索引的纵向合并单元格，**不是**行对齐的。")
w("> 我按药剂名连接，请核「这个药剂 ↔ 这个天数」配得对不对——这是最容易错的一环。\n")

cur = None
for c in B:                                          # 逐张附录 B 卡导出药剂明细表
    if c["crop"] != cur:                             # 作物变化时插入新的作物小节标题
        cur = c["crop"]
        w(f"\n### {cur}（{c['std_no']}）\n")
    w(f"**`{c['card_id']}`** — {c['title']}　`{c['source']['section']} p{c['source']['page']}`\n")
    w("| 药剂 | 剂量/浓度 | 稀释 | 施用方法 | 安全间隔期 | 每茬最多次数 | 施药间隔 | 核对 |")
    w("| --- | --- | --- | --- | --- | --- | --- | :-: |")
    for ch in c["chemicals"]:                        # 逐条药剂导出数值列
        v = lambda x: x if x else "—"                # 空值占位符：缺失字段显示「—」而不是空
        w(f"| {ch['product']} | {ch['dose']} | {v(ch['dilution'])} | {ch['method'][:26]} "
          f"| **{v(ch['pre_harvest_interval'])}** | {v(ch['max_uses_per_season'])} "
          f"| {v(ch['interval_days'])} |  |")        # method 截断 26 字防表格过宽；安全间隔期加粗（重点核对项）
    w("")

w("\n### 附：大蒜 15 条安全间隔期（数值最反常，优先核）\n")
w("| 药剂（原文用字） | 安全间隔期 | 说明 | 核对 |")
w("| --- | --- | --- | :-: |")
# 人工整理的 15 行「药剂↔安全间隔期」速查行（ROWS）：数值反常/极端值的优先核对项
ROWS = [("多菌灵", "52 d", ""), ("腐霉利", "220 d", "全表最大值"),
        ("咯菌腈", "30 d", ""), ("灭蝇胺", "21.7 d", "全表唯一带小数"),
        ("肪硫磷(拌种、灌根)", "63 d", "同药剂两种用法"),
        ("二甲戊乐灵", "100 d", ""), ("敌敌畏", "21 d", ""),
        ("肪硫磷(喷雾)", "28 d", "同药剂两种用法"),
        ("溴氰菊酯", "14 d", ""), ("哒螨灵", "14 d", ""),
        ("代森锰锌", "43 d", ""), ("杀螟丹", "27 d", ""),
        ("吡虫啉", "45 d", ""), ("噻虫嗪", "18 d", ""),
        ("异菌脲", "200 d", "次大值")]
for k, v, note in ROWS:                              # k=药剂名, v=间隔期, note=备注说明
    w(f"| {k} | **{v}** | {note} |  |")
w("\n**落灭津无对应安全间隔期** —— 原表清单里没有它（清单 15 条覆盖 14 个药剂）。")
w("已在卡内记 `missing_fields`，请确认原表确实没有，别让我补。\n")

# ------------------------------------------------ 2. 正文高危
w("\n---\n\n## 二、正文 28 张高危卡 —— 归类核对\n")
w("核对两点：① 该不该算高危（有没有过度/遗漏）② `knowledge_type` 分类对不对。\n")
w("| card_id | 条款 | 内容摘要 | knowledge_type | 高危原因 | 该高危? | 分类对? |")
w("| --- | --- | --- | --- | --- | :-: | :-: |")
for c in O:
    if not c["is_high_risk"]:                        # 只导出高危卡
        continue
    w(f"| `{c['card_id']}` | {c['source']['section']} | {c['source']['quote'][:32]}… "  # 原文引用截 32 字做摘要
      f"| {c['knowledge_type']} | {'、'.join(c['high_risk_reasons'])} |  |  |")

w("\n### 故意**未**标高危的指针条款（请确认这个判断）\n")
w("这几条只说「防治方案见附录B」，本身不含农药知识，真正内容在附录 B 卡里：\n")
for c in O:
    q = c["source"]["quote"]
    if (not c["is_high_risk"]) and ("见附录B" in q or "见附录 B" in q):
        # 非高危、且原文只是「见附录B」的指针条款——列出供人工确认这个不标高管的判断
        w(f"- `{c['card_id']}` {c['source']['section']}：{q[:46]}…")   # 引用截 46 字

w("\n### 判定规则说明（便于你判断规则本身对不对）\n")
w("- 高危上下文 = 各标准自己的「化学防治」小节 ∪「劳动保护」小节 ∪ 农药采购/贮藏/剩余农药条款（4.1.x）")
w("- 条款号**不能硬编码**：黄瓜化学防治是 `6.6`，辣椒是 `6.3.4`，大蒜是 `6.3.3`，脚本按正文里的小节标题定位")
w("- 黄瓜 `3.2.2` 基地仓库原本被误标（文本提到「防护服、急救箱」），已改为非高危——")
w("  若你认为「仓库配备急救箱」也该提示农户，告诉我改回\n")

# ------------------------------------------------ 3. 附录 A vision
w("\n---\n\n## 三、附录 A 的 11 张视觉转录卡 —— 字段名核对\n")
w("这些页是字形码页/旋转 90° 横向排版，字段名全靠视觉转录。")
w("对照 `logs/_verify_png/appA_*.png` 核，重点看**有没有漏列、错字**。\n")
n = 0                                                # vision 通道卡片计数（小节序号）
for c in A:
    if c["source"]["channel"] != "vision":           # 只导出视觉转录（不可文本抽取）的卡
        continue
    n += 1
    f = c["record_form"]
    w(f"### {n}. `{c['card_id']}` — {f['table_no']} {f['table_name']}　`p{c['source']['page']}`\n")
    if f["header_items"]:
        w(f"- **抬头项**：{'、'.join(f['header_items'])}")
    w(f"- **列**（{len(f['columns'])} 项）：{' ｜ '.join(f['columns'])}")
    if f["footnote"]:
        w(f"- **表注**：{f['footnote']}")
    w("- 核对：☐ 字段无漏　☐ 无错字　☐ 列数与原图一致\n")   # 预置的三项人工核对勾选项

w("\n> 注：附录 A 三份标准的**表号→表名映射不一致**（如「生产基地田间农事活动记录表」")
w("> 在黄瓜是 A.12、辣椒/大蒜是 A.7），且**表 A.1 字段也不同**（黄瓜两列 vs 辣椒/大蒜三列）。")
w("> 卡片已按各标准实录，未做统一——请一并确认这个口径。\n")

txt = "\n".join(L)                                   # 行缓冲拼接成完整 Markdown
dst = os.path.join(P, "人工抽检清单.md")
open(dst, "w", encoding="utf-8").write(txt)
print(f"已写入 {dst}（{len(txt)} 字符）")
print(f"附录B {len(B)} 张 / 正文高危 {sum(1 for c in O if c['is_high_risk'])} 张 / "
      f"附录A vision {sum(1 for c in A if c['source']['channel']=='vision')} 张")  # 三块卡数汇总
