"""批次 39：站点级页脚样板覆盖 + 正文区标记专项测试。

背景：批次 38 只重导了 document id=1，同站点的其他页面（doc7 司法解释一、
doc8 调解仲裁法）仍是修复前的旧块，库里继续带「责任编辑 / 总机 / 版权所有 /
京ICP备」；中国政府网页面（doc11）还有「主办单位 / 扫一扫 / 【打印】」等
正文工具条与页脚链接区。修复分三层：
1. parser 正文区容器标记从 zoom 扩到 UCAP-CONTENT（中国政府网），并把
   子串匹配收紧为完整标记匹配（`class="imagezoom"` 曾被误当作正文区）；
2. cleaner 页脚特征扩到站点运营方署名/来源标注/移动端引导，并新增
   「链接区标题 + 紧随链接标题行」的有界处理；
3. 全文工具条控件标签【打印】【我要纠错】按白名单剔除。

红线（必须保留）：法规结构标题（第一章/总则/第五条）、典型案例的正文小标题
（【基本案情】【裁判结果】【典型意义】）不得被样板规则误删。

fixture 取自真实采集页面（`data/labor_law_raw/` 的逐字片段，非手抄），
断言写的是"这两张真实页面应有的结果"，不是当前实现的输出。
"""

from pathlib import Path

from app.ingest.cleaner import clean_text
from app.ingest.parser import parse_document

FIXTURES = Path(__file__).parent / "fixtures"


def _parse_fixture(name: str) -> str:
    """解析真实页面 fixture 片段，返回清洗后的正文。"""
    return parse_document(FIXTURES / name).content


# ---------- fixture 1：法院网典型案例页（doc1 同模板） ----------

def test_case_page_body_and_case_headings_kept() -> None:
    """案例页正文、目录与案例小标题必须保留（小标题是案例结构，不是控件标签）。"""
    text = _parse_fixture("court_case_page.html")
    for expected in ("目录", "案例一", "【基本案情】", "【裁判结果】", "【典型意义】"):
        assert expected in text, f"案例页缺失正文结构：{expected}"


def test_case_page_footer_and_breadcrumb_removed() -> None:
    """案例页页脚版权/备案/校验串与页头面包屑必须清掉。"""
    text = _parse_fixture("court_case_page.html")
    for banned in (
        "所在位置", "字号", "责任编辑", "总机", "举报电话", "传真",
        "版权所有", "京ICP备", "京公网安备", "11040102700145号",
        "f68e0d47",
    ):
        assert banned not in text, f"案例页残留样板：{banned}"


# ---------- fixture 2：法院网司法解释页（doc7 同模板） ----------

def test_interpretation_first_and_last_article_kept() -> None:
    """司法解释首条与末条必须完整保留（末条紧邻页脚，最易被连带删掉）。"""
    text = _parse_fixture("court_interpretation_page.html")
    assert "第一条" in text
    assert "第五十四条本解释自2021年1月1日起施行。" in text
    assert "法释〔2020〕26号" in text


def test_interpretation_footer_removed() -> None:
    """司法解释页页脚（责任编辑/总机/版权/备案/校验串）必须清掉。"""
    text = _parse_fixture("court_interpretation_page.html")
    for banned in (
        "责任编辑", "韩绪光", "总机", "举报电话", "版权所有",
        "京ICP备", "11040102700145号", "f066b984", "所在位置", "字号",
    ):
        assert banned not in text, f"司法解释页残留样板：{banned}"


def test_interpretation_link_area_titles_removed() -> None:
    """「相关链接」区的推荐链接标题属站点导航，不进正文。"""
    text = _parse_fixture("court_interpretation_page.html")
    assert "相关链接" not in text


# ---------- fixture 3：中国政府网页面（doc11 同模板） ----------

def test_gov_page_body_kept() -> None:
    """中国政府网页面：标题、公布信息与首末条必须保留。"""
    text = _parse_fixture("gov_news_page.html")
    for expected in ("工资支付暂行规定", "劳部发〔1994〕489号", "第一条", "第二十条"):
        assert expected in text, f"政府网页面缺失正文：{expected}"


def test_gov_page_toolbar_and_footer_removed() -> None:
    """中国政府网正文工具条、稿件来源标注与页脚链接/署名区必须清掉。"""
    text = _parse_fixture("gov_news_page.html")
    for banned in (
        "【打印】", "【我要纠错】", "【关闭窗口】", "【字体", "扫一扫在手机打开",
        "主办单位", "运行维护单位", "中文域名", "网站标识码",
        "京ICP备", "京公网安备", "国务院客户端", "来源：", "微博、微信",
    ):
        assert banned not in text, f"政府网页面残留样板：{banned}"


# ---------- 正文区标记：不得被子串误命中 ----------

def test_imagezoom_class_is_not_marker(tmp_path: Path) -> None:
    """`class="imagezoom"` 是图片放大组件，不是正文区。

    修复前的子串匹配会把它当成 zoom 正文区，只收到组件内的说明文字、
    丢掉整篇正文。这里让组件内文字足够长（>100 字，绕过"正文过短回退"
    的保护），确保真出问题时测试会红。
    """
    filler = "图片放大组件内的辅助说明文字，与法规正文无关。" * 8
    html = (
        "<html><body>"
        f'<div class="imagezoom"><p>{filler}</p></div>'
        "<p>第一条 真正的正文内容必须被保留。</p>"
        "<p>第二条 本规定自公布之日起施行。</p>"
        "</body></html>"
    )
    path = tmp_path / "imagezoom.html"
    path.write_text(html, encoding="utf-8")
    text = parse_document(path).content
    assert "第一条 真正的正文内容必须被保留。" in text
    assert "第二条 本规定自公布之日起施行。" in text


def test_main_container_matched_as_whole_token() -> None:
    """真实页面写法 `class="txt_txt" id="zoom"` 必须命中正文区，容器外文本不进来。

    注：容器内文本需 >100 字，否则会走「正文过短、回退全页」的保护分支。
    """
    body = (
        "<p>第一条 用人单位应当依法建立和完善劳动规章制度，保障劳动者享有劳动权利、"
        "履行劳动义务，并就劳动报酬、工作时间、休息休假等事项作出规定。</p>"
        "<p>第二条 本规定自公布之日起施行，此前发布的规定与本办法不一致的，以本办法为准。</p>"
    )
    html = (
        '<html><body><div class="outer">页头不该进正文</div>'
        f'<div class="txt_txt big" id="zoom">{body}</div>'
        "</body></html>"
    )
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", encoding="utf-8", delete=False
    ) as fh:
        fh.write(html)
        name = fh.name
    try:
        text = parse_document(Path(name)).content
        assert "第一条 用人单位应当依法建立和完善劳动规章制度" in text
        assert "第二条 本规定自公布之日起施行" in text
        assert "页头不该进正文" not in text
    finally:
        Path(name).unlink(missing_ok=True)


# ---------- 链接区处理：有界、不吞正文 ----------

def test_link_area_window_stops_at_article_line() -> None:
    """链接区标题后的链接标题行被剔除，遇到条文行立即恢复收录。"""
    text = clean_text(
        "相关链接：\n某推荐链接标题\n第一条 为了规范用工，制定本规定。"
    )
    assert "某推荐链接标题" not in text
    assert "第一条 为了规范用工，制定本规定。" in text


def test_link_area_window_is_bounded() -> None:
    """链接区跳跃行数有上限，超窗口的正常行必须保留。"""
    lines = ["相关链接："] + [f"推荐链接{i}" for i in range(10)]
    lines.append("这是超窗口之后应当保留的正常正文行")
    text = clean_text("\n".join(lines))
    assert "这是超窗口之后应当保留的正常正文行" in text


def test_case_body_headings_not_treated_as_control_labels() -> None:
    """【基本案情】类案例小标题不是控件标签，必须保留（批次 39 试跑曾误删）。"""
    text = clean_text("【基本案情】\n原告于2024年3月入职。\n【裁判结果】\n法院驳回上诉。")
    for expected in ("【基本案情】", "【裁判结果】"):
        assert expected in text


def test_toolbar_control_label_removed() -> None:
    """正文工具条控件标签整行剔除。"""
    text = clean_text("第一条 正文。\n【打印】\n【我要纠错】\n【字体：大 中 小】\n第二条 正文。")
    for banned in ("【打印】", "【我要纠错】", "【字体"):
        assert banned not in text
    assert "第一条 正文。" in text and "第二条 正文。" in text
