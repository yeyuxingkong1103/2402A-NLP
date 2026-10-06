# 律师卡片是「演示数据」——技术方案 6.4 定为随机生成，正式运营前必须换成
# 真实在册名录（风险 11：否则构成虚假宣传）。所以这里最要紧的两条断言是：
# ①DEMO_MODE 下必须带「示例数据」标注；②不产出任何真实联系方式。
# 排序规则也有断言：FR-9.5 / AC-21 要求「无付费插位」，排序只能由领域匹配决定。
import random

from app.recommend.cause import GENERIC_FIELD
from app.recommend.lawyers import (
    CONTACT_HINT, DEMO_MODE, DEMO_NOTE, GIVEN_CHARS, ORG_PREFIX, ORG_SUFFIX,
    SURNAMES, recommend,
)


def test_returns_requested_count():
    assert len(recommend("劳动法", 3, random.Random(1))) == 3
    assert len(recommend("劳动法", 1, random.Random(1))) == 1
    # 默认张数也是接口块写死的契约：不传 n 时给 3 张，下游正是靠默认值取卡
    assert len(recommend("劳动法")) == 3


def test_cards_carry_all_required_fields():
    """FR-9.3：姓名、执业机构、专业领域、联系方式获取方式，四项缺一不可。"""
    # 约定文案要钉到字面量：只写「等于常量」时，把常量改成手机号也全绿 ——
    # 演示期不得产出任何真实联系方式（风险 11）。卡片侧按「每一张」钉：其余
    # 断言都只看首张，只钉单张的话「只有第一张是对的」这类坏法能溜过去
    assert CONTACT_HINT == "通过平台留言获取"
    card = recommend("劳动法", 1, random.Random(2))[0]
    assert card["name"]
    assert card["org"].endswith("律师事务所")
    assert card["field"] == "劳动法"
    assert card["contact_hint"] == CONTACT_HINT
    cards = recommend("劳动法", 3, random.Random(2))
    assert all(c["contact_hint"] == "通过平台留言获取" for c in cards)


def test_demo_note_is_present_while_demo_mode():
    """技术方案 6.4 硬规矩：演示期必须能一眼看出是示例数据。"""
    assert DEMO_MODE is True
    # 标注同样要钉字面量，与 DEMO_MODE 那条互相独立：只钉「卡片用的是同一常量」
    # 时，把常量清空成 "" 仍全绿 —— 标注一旦消失，卡片看起来就像真实推荐，
    # 正是风险 11（虚假宣传）的失败方向。卡片侧按「每一张」钉：其余断言都只看
    # 首张，只钉单张的话「只有第一张带标注」这类坏法能溜过去
    assert DEMO_NOTE == "示例数据"
    card = recommend("劳动法", 1, random.Random(3))[0]
    assert card["demo"] is True
    assert card["note"] == DEMO_NOTE
    cards = recommend("劳动法", 3, random.Random(3))
    assert all(c["demo"] is True for c in cards)
    assert all(c["note"] == "示例数据" for c in cards)


def test_card_exposes_only_the_agreed_fields():
    """卡片字段集固定为这六项 —— 将来若有人塞进手机号 / 邮箱之类的真实
    联系方式，这条会红。演示期不得产出任何真实联系方式（风险 11）。"""
    # 按「每一张」查：只查首张的话，给非首张卡偷偷加联系方式字段查不出来
    cards = recommend("劳动法", 3, random.Random(5))
    agreed = {"name", "org", "field", "contact_hint", "demo", "note"}
    assert all(set(c) == agreed for c in cards)


def test_generic_field_still_returns_cards():
    """案由认不出时领域为「通用」，卡片照常展示（设计 §七：不报错）。"""
    cards = recommend(GENERIC_FIELD, 2, random.Random(4))
    assert len(cards) == 2
    assert all(c["field"] == GENERIC_FIELD for c in cards)


def test_same_seed_gives_same_order_and_content():
    """排序与内容只由 (field, seed) 决定 —— 没有任何外部输入（如付费）能影响它。
    这是 FR-9.5「无付费插位」在代码层的可测形式。"""
    a = recommend("劳动法", 3, random.Random(7))
    b = recommend("劳动法", 3, random.Random(7))
    assert a == b
    # 上面那句只证「同一输入两次结果相同」—— 而付费插位在代码里恰好也是这个形态：
    # 末尾追加一条确定性再排序（按名单把这批里的某张置顶）照样满足它。所以还要按
    # 实现声明的抽取顺序把同一批卡片重演一遍，顺序与内容逐张对上，再排序无处藏身。
    # 重演必须完整复刻抽取顺序：先 name 的 randint + 取字，再 2 次 org 取字（漏掉
    # 任何一次抽取都会在干净实现上就报红，重演即失效）。下面 6 行是 recommend
    # 生成形状的**手写镜像**：改动生成形状（抽取的次数/顺序/字符集）须同步改此处
    rng = random.Random(7)
    expected = []
    for _ in range(3):
        given = "".join(rng.choice(GIVEN_CHARS) for _ in range(rng.randint(1, 2)))
        expected.append((rng.choice(SURNAMES) + given,
                         rng.choice(ORG_PREFIX) + rng.choice(ORG_PREFIX) + ORG_SUFFIX))
    assert [(c["name"], c["org"]) for c in
            recommend("劳动法", 3, random.Random(7))] == expected


def test_different_fields_are_reflected_in_cards():
    cards = recommend("婚姻家庭", 3, random.Random(9))
    assert all(c["field"] == "婚姻家庭" for c in cards)
