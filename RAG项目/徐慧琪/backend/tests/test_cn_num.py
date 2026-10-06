# 条号转阿拉伯数字是层级还原的基础，边界值必须逐个钉死
import pytest
from app.ingest.cn_num import cn2int


@pytest.mark.parametrize("cn,expected", [
    ("一", 1),
    ("十", 10),
    ("十五", 15),
    ("二十", 20),
    ("九十九", 99),
    ("二百零四", 204),      # 第一编最后一条
    ("四百六十二", 462),    # 第二编最后一条
    ("四百六十三", 463),    # 第三编第一条
    ("一千零三十九", 1039), # 第四编最后一条
    ("一千零四十", 1040),   # 第五编第一条
    ("一千二百六十", 1260), # 全法典最后一条（附则）
])
def test_cn2int_covers_whole_code(cn, expected):
    # 覆盖面刻意取三册的真实边界条号，而不是随便造数
    assert cn2int(cn) == expected
