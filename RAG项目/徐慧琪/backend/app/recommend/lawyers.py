"""示例律师卡片生成。

为什么随机生成（技术方案 6.4 定稿）：不接真实名录，仅用于演示与联调。
**对外正式运营前必须替换为真实在册律师信息**，否则构成虚假宣传（风险 11）——
所以 DEMO_MODE 为真时卡片必须带「示例数据」标注，这条有测试钉住。
"""
from __future__ import annotations

import random

# 演示开关：接入真实名录时改这里，并把 DEMO_NOTE 从卡片上撤下
DEMO_MODE = True
DEMO_NOTE = "示例数据"

# 演示期固定的联系方式获取方式：不产出任何真实联系方式
CONTACT_HINT = "通过平台留言获取"

# 姓名与机构名用字表。只用常见字，避免生成生僻到像乱码的名字
SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦许何吕施张孔曹严华金魏陶姜"
GIVEN_CHARS = "文伟芳娜敏静秀丽强磊洋勇艳杰娟涛明超霞平刚桂英建国民"
ORG_PREFIX = "明理正衡德信公诚瑞达弘正同仁经纶格物"
ORG_SUFFIX = "律师事务所"


def _name(rng: random.Random) -> str:
    """常见姓 + 1~2 个名用字。"""
    given = "".join(rng.choice(GIVEN_CHARS) for _ in range(rng.randint(1, 2)))
    return rng.choice(SURNAMES) + given


def _org(rng: random.Random) -> str:
    return rng.choice(ORG_PREFIX) + rng.choice(ORG_PREFIX) + ORG_SUFFIX


def recommend(field: str, n: int = 3, rng: random.Random | None = None) -> list[dict]:
    """生成 n 张示例律师卡片。

    排序只由领域匹配度决定（此处即「卡片领域 == 请求领域」），
    没有任何外部输入能改变顺序 —— FR-9.5 / AC-21 的「无付费插位」。
    rng 作为参数而非模块全局：测试要能复现同一批卡片。
    """
    rng = rng or random.Random()
    cards = []
    for _ in range(n):
        card = {
            "name": _name(rng),
            "org": _org(rng),
            "field": field,
            "contact_hint": CONTACT_HINT,
            "demo": DEMO_MODE,
        }
        if DEMO_MODE:
            # 标注与卡片同层，渲染方不必再判断一次开关
            card["note"] = DEMO_NOTE
        cards.append(card)
    return cards
