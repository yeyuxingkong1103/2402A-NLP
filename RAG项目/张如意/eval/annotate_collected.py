# -*- coding: utf-8 -*-
"""给采集来的真实问法补 ground truth -> eval/datasets/collect.jsonl（原地更新）

## ⚠️ 这些标注是**我（Claude）标的，不是农户标的，需要人工复核**

真实问法本身是独立的（来自 12316 热线，不是从卡片反推的）——这一点比 auto 评估集强。
但「这个问题该由哪张卡回答」是我读着问句对着语料判断的，**仍有主观成分**。
在复核之前，collected 的召回率只能当**参考值**，不能当验收依据。

## 标注口径

| 情形 | expect_cards | expect_fallback | expect_noplan |
| --- | --- | --- | --- |
| 语料有该病虫害的用药方案卡 | 附录B 卡 | False | False |
| 语料只在正文提到过，无用药方案 | 正文章节卡 | False | **True** |
| 语料完全没有 | [] | True | False |

`expect_noplan=True` 表示：系统应**返回正文内容 + 带「暂无具体用药方案」提示**，
而不是硬兜底——这是 2026-09-20 定稿的口径。

`is_high_risk` 按**问句是否在问用药**来标（不是按卡片是否高危）——
用来分层看「问药」与「问症状/问措施」的表现差异。
"""
import json
import os

# 采集数据集路径：真实问法存放在 eval/datasets/collect.jsonl，本脚本原地更新它
P = os.path.join("eval", "datasets", "collect.jsonl")

# query 前 8 字（定位用） -> (expect_cards, expect_fallback, expect_noplan, is_high_risk, 标注依据)
# 每条标注是一个五元组：
#   expect_cards    期望命中的卡片 id 列表（附录B 卡或正文章节卡）
#   expect_fallback 语料完全没有该病虫害时应硬兜底（返回「暂无」）
#   expect_noplan   语料只在正文提到、无用药方案时应返回正文+「暂无方案」提示
#   is_high_risk    问句是否在问用药（用于分层统计，与卡片是否高危无关）
#   标注依据        人工复核时的判断理由说明
A = {
 "黄瓜叶片有白色粉末": (["GBZ26581-2011-p20-c06"], False, False, True, "症状→黄瓜白粉病，有附录B卡；问怎么办含用药"),
 "黄瓜叶子背面长了灰黑色": (["GBZ26581-2011-p20-c01", "GBZ26581-2011-p20-c02"], False, False, True, "症状→霜霉病，两张卡（发病前预防/发病初期）"),
 "黄瓜苗突然成片倒伏": (["GBZ26581-2011-p6-b6.2.1"], False, True, False, "症状→猝倒病；黄瓜只在 6.2.1 列名，无附录B方案 → 放宽+提示"),
 "黄瓜叶上有多角形黄褐色": (["GBZ26581-2011-p20-c03"], False, False, True, "症状→细菌性角斑病（多角形斑+菌脓），有卡"),
 "大棚黄瓜霜霉病打什么药": (["GBZ26581-2011-p20-c01", "GBZ26581-2011-p20-c02"], False, False, True, "直接问病名"),
 "黄瓜灰霉病用什么药": (["GBZ26581-2011-p20-c05"], False, False, True, "直接问病名"),
 "黄瓜秧中午萎蔫早晚恢复": (["GBZ26581-2011-p6-b6.2.2"], False, True, False, "症状→枯萎病；黄瓜只在 6.2.2 列名，无附录B方案 → 放宽+提示"),
 "黄瓜叶子上有很多弯弯曲曲": (["GBZ26581-2011-p21-c09"], False, False, False, "症状→美洲斑潜蝇（隧道），有卡；问是什么虫非用药"),
 "黄瓜瓜蚜打什么药": (["GBZ26581-2011-p21-c10"], False, False, True, "直接问虫名"),
 "大棚黄瓜叶子上密密麻麻": (["GBZ26581-2011-p20-c07"], False, False, True, "症状→棕榈蓟马，有卡；问怎么防含用药"),
 "黄瓜种子播种前怎么处理": (["GBZ26581-2011-p6-b6.5.6"], False, False, False, "农事→6.5.6 温汤浸种"),
 "听说大棚黄瓜高温闷棚": (["GBZ26581-2011-p6-b6.5.5"], False, False, False, "农事→6.5.5 高温闷棚"),
 "大蒜叶尖发黄枯死": (["GBZ26578-2011-p20-c01", "GBZ26578-2011-p20-c08"], False, False, True, "症状→叶枯病，两张卡（播种前/发病初期）"),
 "大蒜叶枯病打什么药": (["GBZ26578-2011-p20-c01", "GBZ26578-2011-p20-c08"], False, False, True, "直接问病名"),
 "大蒜储存的蒜头和地里植株": (["GBZ26578-2011-p20-c07"], False, False, False, "症状→刺足根螨，有卡；问是什么虫"),
 "蒜苗叶子上有一条条白色": (["GBZ26578-2011-p20-c09"], False, False, False, "症状→豌豆潜叶蝇（白色弯曲斑道），有卡"),
 "大蒜叶上有小白点密密麻麻": (["GBZ26578-2011-p20-c10"], False, False, False, "症状→蓟马，有卡；问是什么虫害"),
 "大蒜灰霉病用什么药": (["GBZ26578-2011-p20-c02"], False, False, True, "直接问病名"),
 "大蒜地里的杂草长得比蒜苗": (["GBZ26578-2011-p20-c04", "GBZ26578-2011-p5-b5.3.3.3"], False, False, True, "杂草防除；附录B 杂草卡 + 5.3.3.3 除草和覆膜（时机）"),
 "蒜田里飞的小黑蛆": (["GBZ26578-2011-p20-c03", "GBZ26578-2011-p20-c05", "GBZ26578-2011-p20-c06"], False, False, True, "症状→种蝇，三张卡（播种前/成虫/幼虫）"),
 "大蒜种蒜播种前怎么消毒": (["GBZ26578-2011-p20-c01", "GBZ26578-2011-p20-c02", "GBZ26578-2011-p20-c03"], False, False, True, "种蒜处理=浸种/拌种；附录B 播种前三张卡的方案即浸种拌种"),
 "这块地连着种了两年大蒜": (["GBZ26578-2011-p6-b6.3.1.2"], False, False, False, "轮作→6.3.1.2 每2~3年与非百合科轮作"),
 "辣椒苗期猝倒病怎么防治": (["GBZ26583-2011-p20-c01"], False, False, True, "直接问病名；辣椒表把猝倒病与立枯病合为一行"),
 "辣椒茎基部或枝条处出现": (["GBZ26583-2011-p6-b6.2"], False, True, False, "症状→疫病；辣椒只在 6.2 主要防治对象列名，无附录B方案 → 放宽+提示"),
 "辣椒叶子皱缩花叶": (["GBZ26583-2011-p6-b6.2"], False, True, False, "症状→病毒病；只在 6.2 / 兼治列名，无附录B方案 → 放宽+提示"),
 "辣椒嫩梢嫩叶发黄卷曲": (["GBZ26583-2011-p20-c02"], False, False, True, "症状→蚜虫，有卡；问打什么药"),
 "辣椒果被虫钻进去蛀食": (["GBZ26583-2011-p20-c04"], False, False, True, "症状→棉铃虫/烟青虫（蛀果），有卡"),
 "辣椒新叶变小发硬": (["GBZ26583-2011-p21-c05"], False, False, False, "症状→茶黄螨，有卡（p21）；问怎么回事非用药"),
 "辣椒田里的甜菜夜蛾": (["GBZ26583-2011-p6-b6.2"], False, True, True, "辣椒无甜菜夜蛾用药方案卡，只在 6.2/兼治/物理防治提到 → 放宽+提示"),
 "辣椒早疫病和晚疫病怎么区分": (["GBZ26583-2011-p20-c03"], False, False, True, "直接问病名+用药；辣椒表两病合一行"),
 "辣椒苗床里小苗成片倒伏": (["GBZ26583-2011-p20-c01"], False, False, True, "症状→猝倒病/立枯病，有卡"),
 "上一茬种辣椒的地能接着种": (["GBZ26583-2011-p6-b6.3.1.2"], False, False, False, "轮作→6.3.1.2 每2~3年与非茄科轮作"),
}


def main():
    """主流程：读取 collect.jsonl -> 按前缀匹配人工标注表 -> 原地写回并打印统计。

    输入：eval/datasets/collect.jsonl（每行一个 JSON 对象，含 query 字段）
    输出：同文件原地更新，每条新增 expect_cards / expect_fallback /
          expect_noplan / is_high_risk / note / annotated_by 字段；
          终端打印命中数、高危数、放宽数、兜底数等统计。
    """
    # 逐行读取 JSONL，跳过空行，解析成字典列表
    rows = [json.loads(l) for l in open(P, encoding="utf-8") if l.strip()]
    hit = miss = 0  # hit=已标注条数，miss=没有匹配到标注的条数
    for r in rows:
        # 用「前缀匹配」定位标注：标注表的 key 是 query 前 8 字左右，
        # 问句往往带后缀（如"…是生病了吗"），所以用 startswith 而不是全等
        key = next((k for k in A if r["query"].startswith(k)), None)
        if not key:
            miss += 1
            print(f"  ✗ 无标注: {r['query'][:40]}")  # 只打印前 40 字，避免刷屏
            continue
        cards, fb, noplan, hr, why = A[key]  # 解包五元组标注
        r["expect_cards"] = cards        # 期望命中卡
        r["expect_fallback"] = fb        # 期望硬兜底标志
        r["expect_noplan"] = noplan      # 期望「正文+暂无方案」提示标志
        r["is_high_risk"] = hr           # 是否问用药（高危分层）
        # 在原 note 后追加标注依据，用全角竖线分隔
        r["note"] = f"{r.get('note','')}｜标注依据：{why}"
        r["annotated_by"] = "claude（待人工复核）"  # 标记标注来源，提醒人工复核
        hit += 1

    # 原地写回：覆盖原 collect.jsonl，每行一条 JSON（ensure_ascii=False 保留中文）
    with open(P, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印标注质量统计，供人工复核前快速把握分布
    n_hr = sum(1 for r in rows if r["is_high_risk"])  # 高危（问用药）问句数
    n_noplan = sum(1 for r in rows if r.get("expect_noplan"))  # 期望放宽+提示的条数
    print(f"\n  标注完成 {hit}/{len(rows)}（未标注 {miss}）")
    print(f"  高危问句 {n_hr} / 非高危 {len(rows)-n_hr}")
    print(f"  需放宽+提示（expect_noplan）{n_noplan} 条")
    print(f"  应硬兜底 {sum(1 for r in rows if r['expect_fallback'])} 条")


if __name__ == "__main__":
    main()
