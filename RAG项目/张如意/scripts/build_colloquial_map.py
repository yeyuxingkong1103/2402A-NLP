# -*- coding: utf-8 -*-
"""构建映射表。输出三份：

    configs/colloquial_map.json   术语归一化：农户口语 -> 标准术语
    configs/intent_map.json       意图识别：问法 -> 检索意图（**与术语归一是两回事**）
    data/processed/口语映射表.md   人工复核用

## 两个独立维度

`evidence` —— **怎么知道的**
| 值 | 含义 |
| --- | --- |
| `corpus` | 依据来自本项目语料（三份国标原文），可复现 |
| `corpus-infer` | 由语料内术语关系推出，非原文直述 |
| `seed` | CLAUDE.md 第六节 3 初版表 |
| `guess` | 农业通用俗称，**本项目语料无依据** |

`status` —— **能不能直接用**
| 值 | 含义 | 检索层行为 |
| --- | --- | --- |
| `verified` | 已人工确认，可用 | 正常归一化 |
| `pending` | 存疑或语料缺口，**不参与归一化** | 见 `pending_reason` |

`pending_reason`
| 值 | 含义 | 检索层行为 |
| --- | --- | --- |
| `no_card` | 术语在语料里但**无独立防治方案卡** | 命中后走「暂无防治方案，请咨询当地农技站」兜底 |
| `ambiguity` | 俗称有歧义，可能指多个对象 | 不自动归一，转追问或按作物区分 |

## 2026-09-17 人工筛定
- 删 6 条：稀释×4 / 稀释倍数×1（国标无此词，映射了无卡可召）+ 叶子发黄（歧义，更像缺素/水涝）
- 迁 13 条到 `intent_map.json`：安全间隔期×8 / 每茬最多使用次数×5
- `小白虫` 拆两条（蓟马 / 白粉虱）
- 其余存疑按要求标 pending
"""
import json
import os
from collections import defaultdict

# ---------------------------------------------------------------- 语料内（可复现）
# ⚠️ **映射是单向的：口语侧 -> 语料里实际有卡的那个词。**
#    早先写了「瓜蚜↔蚜虫」这种双向项，问题是反向那条（`瓜蚜→蚜虫`）把人送往
#    语料里**没有卡**的词，既没用，又让归一化出现 A→B→A 循环、来回抵消。
#    判据：目标词必须在语料里有独立卡（或至少是正文里的规范表述）。
CORPUS_INFER = [
    ("白粉病", "黄瓜白粉病", "黄瓜", "病害", "国标 6.2.2 与表B.1 均作「黄瓜白粉病」，兼治列作「白粉」"),
    ("白粉", "黄瓜白粉病", "黄瓜", "病害", "附录B 兼治列作「白粉」，同一表内即指黄瓜白粉病"),
    ("黄瓜疫病", "疫病", "黄瓜", "病害", "附录B 兼治列作「黄瓜疫病」，6.2.2 作「疫病」"),
    # 「蚜虫」在黄瓜语料里的规范名是「瓜蚜」（有卡）；反向的「瓜蚜→蚜虫」已删
    ("蚜虫", "瓜蚜", "黄瓜", "虫害", "农户说「蚜虫」，黄瓜的规范名是「瓜蚜」（有卡）"),
    # 同理，只保留指向有卡词的方向
    ("潜叶蝇", "美洲斑潜蝇", "黄瓜", "虫害", "6.2.2 列「潜叶蝇」，附录B 列「美洲斑潜蝇」（有卡）"),
    # 2026-09-23 补：大蒜侧「潜叶蝇」此前无条目 → coverage 判「语料中大蒜无此病虫害」硬兜底，
    #   而大蒜 6.2 与表B.1 都写「豌豆潜叶蝇」（有 c09 卡）。同词两条靠 crop 分开：
    #   黄瓜→美洲斑潜蝇（本行上面，靠前定义，crop 为空时它先命中）
    ("潜叶蝇", "豌豆潜叶蝇", "大蒜", "虫害", "大蒜 6.2 与表B.1 均作「豌豆潜叶蝇」（有卡）"),
    # 2026-09-23 人工定档（选 A）：黄瓜语料只有「棕榈蓟马」这一个种，农户说「蓟马」按它召回；
    #   属→种推断，故答案里必须出现具体种名（卡 subtype 即「棕榈蓟马」，模板照卡渲染即满足）。
    #   大蒜侧另立（表B.1 单列「蓟马」），靠 crop 分开。
    ("蓟马", "棕榈蓟马", "黄瓜", "虫害", "黄瓜 6.2.2 与表B.1 均作「棕榈蓟马」（有卡）；人工定档按它召回"),
    ("烟粉虱", "白粉虱", "黄瓜", "虫害", "6.2.2 作「烟粉虱」，附录B 兼治列作「白粉虱」；烟粉虱无独立卡 → 标 pending"),
    ("朱砂叶螨", "茶黄螨", "黄瓜", "虫害", "两者在黄瓜表里同为螨类兼治对象，**非同一物种，仅作同类提示**"),
    ("猝倒病", "猝倒病、立枯病", "辣椒", "病害", "辣椒表B.1 将两病合为一行"),
    ("立枯病", "猝倒病、立枯病", "辣椒", "病害", "同上"),
    ("早疫病", "早疫病、晚疫病", "辣椒", "病害", "辣椒表B.1 将两病合为一行"),
    ("晚疫病", "早疫病、晚疫病", "辣椒", "病害", "同上"),
    ("棉铃虫", "棉铃虫、烟青虫", "辣椒", "虫害", "辣椒表B.1 将两虫合为一行"),
    ("烟青虫", "棉铃虫、烟青虫", "辣椒", "虫害", "同上"),
    ("蒜蛆", "种蝇", "大蒜", "虫害", "蒜区对种蝇幼虫的通行叫法；语料内无此词，但限大蒜无歧义"),
    # 2026-09-23 补：农户说「换茬/连作/重茬」，三份正文的规范表述都是「轮作」
    #   （各作物 6.3.1.2 轮作；俗称本身语料 0 命中，同「蒜蛆→种蝇」）。不限作物——三份都有。
    #   实测反事实：collected 两条换茬问句归一后 b6.3.1.2 都排第 1（0.5678/0.8909 过闸门）；
    #   加意图路由到农业防治反而更差（辣椒池内 0.0759 < 闸门，会变误兜底）。
    ("换茬", "轮作", None, "其他", "三份正文均作「轮作」（6.3.1.2）；俗称语料 0 命中"),
    ("连作", "轮作", None, "其他", "同上；农户常问「连作两年了还能种吗」"),
    ("重茬", "轮作", None, "其他", "同上"),
]

DRUG_ALIAS = [
    ("烯酰吗琳", "烯酰吗啉", "黄瓜", "同一药剂：辣椒作「烯酰吗啉」，黄瓜作「烯酰吗琳」（疑标准笔误）"),
    ("霜脲氰+唑菌酮", "霜脲氰+噁唑菌酮", "辣椒", "复配剂全名须带「噁」"),
    ("肪硫磷", "肟硫磷", "大蒜", "形近字误写"),
    ("辛硫磷", "肟硫磷", "大蒜", "疑似同物——**需农技确认**"),
]

# CLAUDE.md 第六节 3 初版表 —— 2026-09-17 人工逐条定档
# (口语, 标准, 作物, 类别, status, pending_reason, 备注)
SEED = [
    ("起腻虫", "蚜虫", None, "虫害", "verified", None,
     "蚜虫有卡。黄瓜侧再经「蚜虫→瓜蚜」链式归一，故此处留通用"),
    ("叶子发黄", "黄化", None, "其他", "pending", "no_card",
     "「黄化」是症状描述，不是独立病虫害；且更可能是缺素/水涝——人工定档：no_card"),
    ("烂根", "根腐病", None, "病害", "pending", "no_card",
     "根腐病仅作兼治出现，无独立卡——人工定档：no_card"),
    ("白粉", "白粉病", None, "病害", "verified", None,
     "白粉病有卡（黄瓜）。黄瓜侧另有「白粉→黄瓜白粉病」专属条目优先"),
    ("卷叶", "卷叶病/虫害", None, "其他", "pending", "no_card",
     "「卷叶病/虫害」是症状描述且本身是二选一的模糊术语——人工定档：no_card"),
]

# ---------------------------------------------------------------- 人工筛定结果
# (口语, 标准, 作物, 类别, status, pending_reason, 备注)
GUESS = [
    # ---- 虫害 ----
    ("腻虫", "蚜虫", "辣椒", "虫害", "verified", None, "仅辣椒有卡"),
    ("蜜虫", "蚜虫", "辣椒", "虫害", "verified", None, ""),
    ("油虫", "蚜虫", "辣椒", "虫害", "verified", None, ""),
    ("蚜子", "蚜虫", "辣椒", "虫害", "verified", None, ""),
    ("黑蚜", "蚜虫", "辣椒", "虫害", "verified", None, ""),
    ("腻巴虫", "蚜虫", "辣椒", "虫害", "verified", None, ""),
    ("画符虫", "美洲斑潜蝇", "黄瓜", "虫害", "verified", None, "仅黄瓜有卡"),
    ("鬼画符", "美洲斑潜蝇", "黄瓜", "虫害", "verified", None, ""),
    ("串皮虫", "美洲斑潜蝇", "黄瓜", "虫害", "verified", None, ""),
    ("红蜘蛛", "朱砂叶螨", "黄瓜", "虫害", "pending", "ambiguity",
     "也可能指二斑叶螨；且朱砂叶螨无独立卡"),
    ("火龙", "朱砂叶螨", "黄瓜", "虫害", "pending", "no_card", "朱砂叶螨仅作兼治出现，无独立卡"),
    ("小白虫", "蓟马", "大蒜", "虫害", "verified", None, "已拆：另见「小白虫→白粉虱」。两者都可能指，检索需并列"),
    ("小白虫", "白粉虱", None, "虫害", "verified", None, "已拆：与「小白虫→蓟马」并列"),
    ("蓟马子", "蓟马", "大蒜", "虫害", "verified", None, ""),
    ("小白蛾子", "白粉虱", None, "虫害", "verified", None, ""),
    ("钻心虫", "棉铃虫", "辣椒", "虫害", "pending", "ambiguity",
     "钻心虫在玉米上常指玉米螟，辣椒上指棉铃虫——必须按作物区分，不可全局归一"),
    ("棉铃子", "棉铃虫", "辣椒", "虫害", "verified", None, ""),
    ("土蚕", "地下害虫", "大蒜", "虫害", "pending", "no_card", "仅大蒜 6.3.2.2 提到，无独立卡"),
    ("地蚕", "地下害虫", "大蒜", "虫害", "pending", "no_card", "同上"),
    ("拉拉蛄", "蝼蛄", "大蒜", "虫害", "pending", "no_card", "仅大蒜 6.3.2.2 杀虫灯诱杀对象，无独立卡"),
    ("蝼蛄", "蝼蛄", "大蒜", "虫害", "pending", "no_card", "同上"),
    ("土蚕子", "地老虎", "大蒜", "虫害", "pending", "no_card", "仅大蒜 6.3.2.2 杀虫灯诱杀对象，无独立卡"),
    ("切根虫", "地老虎", "大蒜", "虫害", "pending", "no_card", "同上"),
    ("根蛆", "种蝇", "大蒜", "虫害", "verified", None, "限大蒜"),
    # 「蒜蛆」已在 CORPUS_INFER

    # ---- 病害 ----
    ("上白粉", "黄瓜白粉病", "黄瓜", "病害", "verified", None, "仅黄瓜有卡"),
    ("白毛", "黄瓜白粉病", "黄瓜", "病害", "verified", None, ""),
    ("起白霜", "黄瓜白粉病", "黄瓜", "病害", "verified", None, ""),
    ("跑马干", "霜霉病", "黄瓜", "病害", "verified", None, "仅黄瓜有卡"),
    ("黑毛", "霜霉病", "黄瓜", "病害", "verified", None, "注意与「灰毛→灰霉病」区分"),
    ("起黑斑", "霜霉病", "黄瓜", "病害", "verified", None, ""),
    ("灰毛", "灰霉病", None, "病害", "verified", None, "黄瓜·大蒜均有卡"),
    ("烂果", "灰霉病", None, "病害", "verified", None, ""),
    ("烂秧", "菌核病", None, "病害", "pending", "no_card", "菌核病仅作兼治出现，无独立卡"),
    ("根烂", "根腐病", "辣椒", "病害", "pending", "no_card", "根腐病仅作兼治出现，无独立卡"),
    ("倒苗", "猝倒病、立枯病", "辣椒", "病害", "verified", None, "仅辣椒有卡"),
    ("站着死", "猝倒病、立枯病", "辣椒", "病害", "verified", None, ""),
    ("死秧", "疫病", "黄瓜", "病害", "pending", "no_card", "疫病仅作兼治出现，无独立卡"),
    ("萎蔫", "枯萎病", "黄瓜", "病害", "pending", "no_card", "枯萎病仅 6.2.2 列名，无独立卡"),
    ("花叶", "病毒病", None, "病害", "pending", "no_card", "病毒病仅 6.2.2 / 兼治出现，无独立卡"),
    ("卷叶", "病毒病", None, "病害", "pending", "no_card", "同上；卷叶也可能是虫害（蚜虫传毒）"),
    ("叶子卷", "病毒病", None, "病害", "pending", "no_card", "同上"),
    ("皱叶", "病毒病", None, "病害", "pending", "no_card", "同上"),
    ("叶子长斑", "叶枯病", "大蒜", "病害", "verified", None, "仅大蒜有卡"),
    ("叶干", "叶枯病", "大蒜", "病害", "verified", None, ""),

    # ---- 杂草 ----
    ("野草", "杂草", "大蒜", "杂草", "verified", None, "仅大蒜有卡"),
    ("长草", "杂草", "大蒜", "杂草", "verified", None, ""),

    # 「打药/打多少」这 6 条已于 2026-09-17 迁往意图表 —— 见 INTENTS
]

# ---------------------------------------------------------------- 意图表（与术语归一分开）
# 这些不是词汇同义，是**问句意图**：农户问「几天能摘」想要的是安全间隔期字段，
# 不是把「几天能摘」当成「安全间隔期」的同义词。
INTENTS = [
    ("隔几天能摘", "pre_harvest_interval", "安全间隔期", "采收前停药天数"),
    ("多少天能吃", "pre_harvest_interval", "安全间隔期", ""),
    ("几天能摘", "pre_harvest_interval", "安全间隔期", ""),
    ("几天能吃", "pre_harvest_interval", "安全间隔期", ""),
    ("几天能采", "pre_harvest_interval", "安全间隔期", ""),
    ("多久能摘", "pre_harvest_interval", "安全间隔期", ""),
    ("多久能吃", "pre_harvest_interval", "安全间隔期", ""),
    ("几天后能收", "pre_harvest_interval", "安全间隔期", ""),
    ("打几次", "max_uses_per_season", "每茬最多使用次数", "同一茬口最多施药次数"),
    ("喷几遍", "max_uses_per_season", "每茬最多使用次数", ""),
    ("能打几次", "max_uses_per_season", "每茬最多使用次数", ""),
    ("最多打几次", "max_uses_per_season", "每茬最多使用次数", ""),
    ("喷几次", "max_uses_per_season", "每茬最多使用次数", ""),
    # 2026-09-17 由术语表迁入：它们指向的是**栏目/字段**，不是可检索的对象
    ("打药", "chemical_control", "化学防治", "要的是防治方案，不是「化学防治」这个词"),
    ("喷药", "chemical_control", "化学防治", ""),
    ("上药", "chemical_control", "化学防治", ""),
    ("打什么药", "chemical_control", "化学防治", "同上"),
    ("用什么药", "chemical_control", "化学防治", "冒烟测试补：实测「大蒜杂草用什么药」漏识别"),
    ("用啥药", "chemical_control", "化学防治", ""),
    ("打啥药", "chemical_control", "化学防治", ""),
    ("打什么农药", "chemical_control", "化学防治", ""),
    ("用什么农药", "chemical_control", "化学防治", ""),
    ("怎么用药", "chemical_control", "化学防治", ""),
    ("用什么药剂", "chemical_control", "化学防治", ""),
    ("打多少", "dose", "剂量", "要的是 chemicals[].dose 数值"),
    ("用多少", "dose", "剂量", "同上"),
    # 2026-09-17 冒烟测试补：这三类正文 5.x 栽培管理有依据，此前漏了
    ("什么时候施肥", "fertilizer", "施肥", "正文 5.1.3 基肥 / 5.5.4 追肥"),
    ("施什么肥", "fertilizer", "施肥", ""),
    ("怎么施肥", "fertilizer", "施肥", ""),
    ("追肥", "fertilizer", "施肥", ""),
    ("施多少肥", "fertilizer", "施肥", ""),
    ("什么时候浇水", "irrigation", "浇水", "正文 5.5.2 浇水、排涝"),
    ("几天浇一次水", "irrigation", "浇水", ""),
    ("怎么浇水", "irrigation", "浇水", ""),
    ("选什么品种", "variety", "品种选择", "正文 5.4 品种选择"),
    ("什么品种好", "variety", "品种选择", ""),
    ("什么时候种", "planting", "播种", "正文 5.2/5.3 播种"),
    ("怎么播种", "planting", "播种", ""),
    ("什么时候播种", "planting", "播种", ""),
    ("什么时候收", "harvest", "采收", "正文 5.6 采收"),
    ("什么时候采收", "harvest", "采收", ""),
    ("怎么收", "harvest", "采收", ""),
    # 2026-09-20 意图路由补充：这三类要单独识别，否则会召回附录B 的用药方案
    ("防治原则", "control_principle", "6.x 防治原则/农业/生物/物理防治", "问原则不该答用药方案"),
    ("怎么防", "control_principle", "6.x 防治原则/农业/生物/物理防治", ""),
    ("怎么防虫", "control_principle", "6.x 防治原则/农业/生物/物理防治", ""),
    ("怎么防病", "control_principle", "6.x 防治原则/农业/生物/物理防治", ""),
    ("防虫治病", "control_principle", "6.x 防治原则/农业/生物/物理防治", ""),
    ("防病治虫", "control_principle", "6.x 防治原则/农业/生物/物理防治", ""),
    ("按啥顺序来防", "control_principle", "6.x 防治原则/农业/生物/物理防治", ""),
    ("预防为主", "control_principle", "6.x 防治原则/农业/生物/物理防治", ""),
    ("综合防治", "control_principle", "6.x 防治原则/农业/生物/物理防治", ""),
    ("优先用", "control_principle", "6.x 防治原则/农业/生物/物理防治", ""),
    ("主要防治对象", "symptom", "6.2 主要防治对象", ""),
    ("主要病虫害", "symptom", "6.2 主要防治对象", ""),
    ("防哪些病虫害", "symptom", "6.2 主要防治对象", ""),
    ("有哪些病虫害", "symptom", "6.2 主要防治对象", ""),
    ("有什么病虫害", "symptom", "6.2 主要防治对象", ""),
    ("有哪些杂草", "symptom", "6.2 主要防治对象", ""),
    ("有什么补贴", "policy", "政策补贴", "本批语料无政策条款，命中即兜底"),
    ("有没有补贴", "policy", "政策补贴", ""),
    ("政策", "policy", "政策补贴", ""),
    # 2026-09-20 粒度错位回归补：这两类的问法不带「打什么药」，会被当成
    # 普通检索 → 附录B 胜出。实际该查的是正文的施药规范 / 物理防治条款。
    # 必须比「打药」(2 字) 长，否则同长时先定义的 chemical_control 会赢
    ("哪些规矩", "chemical_rules", "正文 化学防治小节（施药规范）", "「打药得按哪些规矩」问规范不是问药"),
    ("有啥规矩", "chemical_rules", "正文 化学防治小节（施药规范）", ""),
    ("什么规矩", "chemical_rules", "正文 化学防治小节（施药规范）", ""),
    ("按哪些要求", "chemical_rules", "正文 化学防治小节（施药规范）", ""),
    ("施药要求", "chemical_rules", "正文 化学防治小节（施药规范）", ""),
    ("施药规范", "chemical_rules", "正文 化学防治小节（施药规范）", ""),
    # 用**问句里实际出现的词**：「闷上两个钟头」「热水烫一下」——
    # 按书面词「闷棚」「烫种」写会一条都匹配不上（实测）
    ("闷", "physical_control", "正文 物理防治小节", "如 6.5.5 高温闷棚"),
    ("烫", "physical_control", "正文 物理防治小节", "如 6.5.6 温汤浸种"),
    ("浸种", "physical_control", "正文 物理防治小节", ""),
    ("温汤", "physical_control", "正文 物理防治小节", ""),
]


def build():
    """汇总各来源（CORPUS_INFER / DRUG_ALIAS / SEED / GUESS）的映射条目，
    组装术语表与意图表。

    返回 (rows, intents)：rows 为术语归一化条目列表（每条含 evidence /
    status / pending_reason 字段），intents 为意图表条目列表。
    """
    rows, seen = [], set()

    def add(spoken, canon, crop, cat, ev, status="verified", reason=None, note=""):
        """添加一条术语映射。

        参数：spoken=口语词，canon=标准术语，crop=作物（None 表示通用），
        cat=类别，ev=evidence（怎么知道的），reason=pending_reason，note=备注。
        """
        k = (spoken, canon, crop)   # 去重键：口语词 + 标准术语 + 作物
        if k in seen:               # 已存在则跳过（先定义的条目优先生效）
            return
        seen.add(k)
        rows.append({"spoken": spoken, "canonical": canon, "crop": crop, "category": cat,
                     "evidence": ev, "status": status, "pending_reason": reason, "note": note})

    for sp, ca, crop, cat, note in CORPUS_INFER:
        # 烟粉虱无独立卡 -> pending
        st, rs = ("pending", "no_card") if ca == "烟粉虱" else ("verified", None)
        add(sp, ca, crop, cat, "corpus-infer", st, rs, note)
    for canon, alias, crop, note in DRUG_ALIAS:
        st = "pending" if "需农技确认" in note else "verified"
        add(alias, canon, crop, "药剂", "corpus", st, "ambiguity" if st == "pending" else None, note)
    for sp, ca, crop, cat, st, rs, note in SEED:
        add(sp, ca, crop, cat, "seed", st, rs, note)
    for sp, ca, crop, cat, st, rs, note in GUESS:
        add(sp, ca, crop, cat, "guess", st, rs, note)

    intents = [{"pattern": p, "intent": i, "canonical_field": f, "note": n}
               for p, i, f, n in INTENTS]
    return rows, intents


if __name__ == "__main__":
    # 生成术语表与意图表，并写出两个 JSON 配置文件
    rows, intents = build()
    os.makedirs("configs", exist_ok=True)
    json.dump(rows, open("configs/colloquial_map.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)   # 输出 1：术语归一化表
    json.dump(intents, open("configs/intent_map.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)   # 输出 2：意图识别表

    # 以下统计各维度分布，用于生成人工复核用的 Markdown 报告
    ver = [r for r in rows if r["status"] == "verified"]
    pen = [r for r in rows if r["status"] == "pending"]
    by_reason = defaultdict(list)   # pending 条目按 pending_reason 分组
    for r in pen:
        by_reason[r["pending_reason"]].append(r)
    by_ev = defaultdict(int)        # 按 evidence 维度统计条数
    for r in rows:
        by_ev[r["evidence"]] += 1

    L = ["# 农户口语 → 标准术语 映射表\n",
         "> 术语归一化：`configs/colloquial_map.json`　意图识别：`configs/intent_map.json`",
         "> 生成：`scripts/build_colloquial_map.py`　自测：`scripts/test_recall.py`\n",
         "## 两个维度：怎么知道的（evidence）× 能不能用（status）\n",
         "| evidence | 含义 | 条数 |", "| --- | --- | ---: |"]
    for k, d in [("corpus", "语料原文，可复现"), ("corpus-infer", "语料内术语关系推出"),
                 ("seed", "CLAUDE.md 初版表"), ("guess", "农业通用俗称，本项目语料无依据")]:
        L.append(f"| `{k}` | {d} | {by_ev[k]} |")
    L.append("\n| status | 含义 | 条数 | 检索层行为 |")
    L.append("| --- | --- | ---: | --- |")
    L.append(f"| `verified` | 已人工确认 | {len(ver)} | 正常归一化 |")
    L.append(f"| `pending` | 存疑或语料缺口，**不参与归一化** | {len(pen)} | 见 pending_reason |")
    L.append(f"\n**术语表合计 {len(rows)} 条；意图表 {len(intents)} 条**\n")
    L.append("| pending_reason | 含义 | 条数 | 检索层行为 |")
    L.append("| --- | --- | ---: | --- |")
    L.append(f"| `no_card` | 术语在语料里但**无独立防治方案卡** | {len(by_reason['no_card'])} "
             "| 命中后走「暂无防治方案，请咨询当地农技站」兜底 |")
    L.append(f"| `ambiguity` | 俗称有歧义，可能指多个对象 | {len(by_reason['ambiguity'])} "
             "| 不自动归一，转追问或按作物区分 |\n")
    L.append("> ⚠️ `pending: no_card` **不是映射错了，是语料缺口**。")
    L.append("> 这三份国标里 `病毒病`/`根腐病`/`枯萎病`/`疫病`/`菌核病`/`地下害虫` 等")
    L.append("> 只在「主要防治对象」或「兼治」里列了名，**没有对应的附录B 用药方案**。")
    L.append("> 农户问到这些，正确行为是答「暂无防治方案」并引导咨询农技站，而不是硬凑一个方案。\n")

    for title, sub in [("一、已确认（verified）", ver), ("二、待定（pending）", pen)]:
        L.append(f"\n---\n\n## {title}（{len(sub)} 条）\n")
        L.append("| 口语 | → 标准术语 | 作物 | 类别 | evidence | pending_reason | 备注 |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in sorted(sub, key=lambda x: (x["category"], x["canonical"])):
            L.append(f"| {r['spoken']} | **{r['canonical']}** | {r['crop'] or '通用'} | {r['category']} "
                     f"| `{r['evidence']}` | {r['pending_reason'] or '—'} | {r['note']} |")

    L.append(f"\n---\n\n## 三、意图表（{len(intents)} 条，**不是术语归一**）\n")
    L.append("这些是**问句意图**：农户问「几天能摘」要的是安全间隔期这个**字段**，")
    L.append("不是把「几天能摘」当成「安全间隔期」的同义词。混进术语表会让检索层逻辑变脏。\n")
    L.append("| 问法 | intent | 对应字段 | 含义 |")
    L.append("| --- | --- | --- | --- |")
    for it in intents:
        L.append(f"| {it['pattern']} | `{it['intent']}` | {it['canonical_field']} | {it['note']} |")

    L.append("\n---\n\n## 四、2026-09-17 人工筛定记录\n")
    L.append("**删 6 条**：`稀释`×4、`稀释倍数`×1（国标无此词，映射了无卡可召）、"
             "`叶子发黄→病毒病`（更像缺素/水涝，歧义过大）\n")
    L.append("**迁 19 条到意图表**：")
    L.append("- 安全间隔期×8、每茬最多使用次数×5")
    L.append("- `打药`/`喷药`/`上药`/`打什么药`→`chemical_control`、`打多少`/`用多少`→`dose`（6 条）")
    L.append("  理由：「化学防治」是章节名、「剂量」是字段名，**都不是可检索的对象**，")
    L.append("  农户问的是意图不是术语同义。混在术语表里会让检索层逻辑变脏。\n")
    L.append("**`小白虫` 拆两条**：→蓟马、→白粉虱（两者都可能指，检索需并列）\n")
    L.append("**歧义 3 条标 `pending: ambiguity`**：")
    L.append("- `钻心虫→棉铃虫`：玉米上常指玉米螟，必须按作物区分")
    L.append("- `红蜘蛛→朱砂叶螨`：也可能指二斑叶螨")
    L.append("- `辛硫磷→肟硫磷`：疑似同物，需农技确认\n")
    L.append("**CLAUDE.md 初版表 5 条逐条定档**：")
    L.append("- 保留（verified）：`起腻虫→蚜虫`、`白粉→白粉病` —— 标准术语有卡")
    L.append("- 标 pending:no_card：`叶子发黄→黄化`、`烂根→根腐病`、`卷叶→卷叶病/虫害`")
    L.append("  —— 前两者是**症状描述**不是独立病虫害，且根腐病无独立卡；")
    L.append("  后者本身是「卷叶病/虫害」这种二选一的模糊术语\n")
    L.append("**其余存疑按语料缺口标 `pending: no_card`** —— 这些不是映射错，是语料缺口\n")

    # 写出人工复核用的 Markdown 映射表（逐行拼 L 列表后一次写入）
    open("data/processed/口语映射表.md", "w", encoding="utf-8").write("\n".join(L))

    print(f"术语表 {len(rows)} 条：verified {len(ver)} / pending {len(pen)}"
          f"（no_card {len(by_reason['no_card'])}、ambiguity {len(by_reason['ambiguity'])}）")
    print(f"意图表 {len(intents)} 条")
    print(f"evidence 分布: {dict(by_ev)}")
