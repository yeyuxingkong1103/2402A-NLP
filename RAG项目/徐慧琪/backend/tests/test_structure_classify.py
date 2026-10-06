# 层级判定是入库链路的核心。用例全部取自三册的真实产物行，不是构造的边界
import pytest
from app.ingest.structure import classify


@pytest.mark.parametrize("line,kind,num", [
    # 常规情形：三册各自的真实行
    ("第一条为了保护民事主体的合法权益，调整民事关系…", "条", 1),
    ("## 第一编 总则", "编", 1),
    ("## 第一章 基本规定", "章", 1),
    ("#### 第一节 一般规定", "节", 1),
    ("第一分编 通 则", "分编", 1),
    ("第一千零四十条 本编调整因婚姻家庭产生的民事关系。", "条", 1040),
    # 坑 1：MinerU 把「条」误标成 heading（上册 10 条中招：210/261/348/366/373/393/395/400/401/412）
    ("## 第二百一十条不动产登记，由不动产所在地的登记机构办理。", "条", 210),
    # 坑 2：MinerU 把标记转义（中册经 md→docx 往返后出现）
    ("\\## 第三编 合同", "编", 3),
    # 坑 3：docx 路径的标题带粗体（下册）
    ("# **第五编 婚姻家庭**", "编", 5),
    # 坑 4：标题被包进引用块
    ("> ## 第三编 合同", "编", 3),
])
def test_classify_known_cases(line, kind, num):
    got_kind, got_num, _ = classify(line)
    assert (got_kind, got_num) == (kind, num)


def test_blockquote_metadata_is_not_a_level_line():
    # 头部元信息以 > 开头但不是标题。若被误判成层级行，后面所有条的 path
    # 都会带上「本册范围」这类元信息——这是实测产物里真实存在的行
    kind, num, _ = classify("> 本册范围：第三编 合同、第四编 人格权")
    assert kind is None and num is None


def test_fenbian_not_swallowed_by_bian():
    # 「第一分编」若先判「编」，会被 ^第X编 提前吃掉并误判成「第一编」
    kind, num, _ = classify("第一分编 典型合同")
    assert (kind, num) == ("分编", 1)


def test_non_level_line_returns_none():
    # 款/项/正文不能被误判成层级行，否则整册结构会散架
    kind, num, text = classify("婚姻、收养、监护等有关身份关系的协议，适用…")
    assert kind is None and num is None
    assert text.startswith("婚姻、收养")


def test_normalization_strips_all_lead_chars():
    # 归一化文本不应残留前导标记，供后续按条号切正文
    _, _, text = classify("## 第三百四十八条通过招标、拍卖…")
    assert text.startswith("第三百四十八条")


def test_normalization_strips_trailing_bold_markers():
    # 粗体标记是成对的，只剥前导会留下尾部 `**`，污染 path（FR-7.3 要展示路径）
    _, _, text = classify("# **第五编 婚姻家庭**")
    assert text == "第五编 婚姻家庭"
