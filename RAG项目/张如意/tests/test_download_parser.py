# -*- coding: utf-8 -*-
"""采集层离线单测：不联网，只测 parse_list / safe_filename / 排除逻辑。"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = "AB" * 16                                  # 模拟 onclick="showInfo('...')" 里的哈希参数

# 按文件路径动态加载被测模块（src/ingest/download_gb_standards.py），不联网
spec = importlib.util.spec_from_file_location(
    "dl", ROOT / "src/ingest/download_gb_standards.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


def _row(badge="2021-01-01", name="菠菜生产技术规范", stdno="GB/Z 26573-2011"):
    """构造标准目录列表页的一行 <tr> HTML：单元格依次为序号 / 标准号链接 / 发布日期徽标 / 名称链接。"""
    return (f"<tr><td>1</td><td><a onclick=\"showInfo('{H}');\">{stdno}</a></td>"
            f"<td>{badge}</td><td class=\"mytxt\"><a onclick=\"showInfo('{H}');\">{name}</a></td></tr>")


def test_parse_list_正常行():
    """正常行要解析出标准号、名称；非「采」标、未命中排除词时 cai / excluded 均为 False。"""
    it, = dl.parse_list(_row())
    assert it["stdno"] == "GB/Z 26573-2011"
    assert it["name"] == "菠菜生产技术规范"
    assert it["cai"] is False and it["excluded"] is False


def test_parse_list_采标标记():
    """徽标单元格出现「采」字样时，cai 必须标记为 True（指导性技术文件采标）。"""
    assert dl.parse_list(_row(badge='<span class="label label-warning">采</span>'))[0]["cai"] is True


def test_parse_list_畸形行不崩():
    """缺字段的残缺行解析结果为空列表，不能抛异常拖垮整页采集。"""
    assert dl.parse_list("<tr><td>1</td></tr>") == []


def test_排除关键词可配置():
    """排除词默认不生效；显式传入时命中该词的标准被标记 excluded，未命中的不受影响。"""
    assert dl.parse_list(_row(name="实验动物 引种技术规程"))[0]["excluded"] is False   # 默认不排除
    assert dl.parse_list(_row(name="实验动物 引种技术规程"), exclude=("实验动物",))[0]["excluded"] is True
    assert dl.parse_list(_row(name="中医技术操作规范 儿科"), exclude=("实验动物",))[0]["excluded"] is False


def test_列表URL按类型与ICS拼装():
    """列表页 URL 模板要按页码参数与 ICS 分类号正确拼装出查询参数。"""
    url = dl.LIST_URL.format(p1="3", ics="11")
    assert "p.p1=3" in url and "p.p6=11" in url and "pageSize=50" in url


def test_翻页URL替换页码():
    """翻页要替换（而非追加）page 参数：已有 page=1 时换成目标页码，无参数或带其他参数时正确追加。"""
    base = dl.LIST_URL.format(p1="3", ics="11")
    assert "page=3" in dl.page_url(base, 3) and "page=1" not in dl.page_url(base, 3)
    assert dl.page_url("https://x.cn/list?a=1", 2).endswith("&page=2")
    assert dl.page_url("https://x.cn/list?", 5).endswith("page=5")


def test_safe_filename_去非法字符保留名称():
    """文件名清洗：斜杠等 Windows 非法字符替换为下划线；冒号必须被去掉；合法名称原样保留。"""
    assert dl.safe_filename("GB/Z 5169.50-2026", "标/准") == "GB_Z 5169.50-2026 标_准"
    assert ":" not in dl.safe_filename("GB/T 1-2020", "第1部分:术语")
    assert dl.safe_filename("GB/Z 26589-2011", "洋葱生产技术规范") == "GB_Z 26589-2011 洋葱生产技术规范"
