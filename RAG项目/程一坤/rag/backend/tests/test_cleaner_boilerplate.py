"""批次 38：站点页脚样板过滤 + TRS 正文区提取专项测试。

背景：court.gov.cn 老式 TRS 页面的页头（面包屑「所在位置：」「字号：」）与
页脚（版权所有 / 公安备案号 / 京ICP备 / 32 位站点统计校验串）都是普通 div，
HTML 解析器只忽略语义标签拦不住，整页样板被当作正文入库（document id=1
的 parent 块混入页脚）。修复分两层：
1. parser 优先提取 TRS 正文区容器（id/class 含 zoom），未命中回退全页；
2. cleaner 增加页脚样板行过滤，兜底无正文区标记的页面。

红线：法规结构标题（第一章/总则/第五条）不得被样板规则误删。
"""

from pathlib import Path

from app.ingest.cleaner import clean_text
from app.ingest.parser import parse_document


# ---------- cleaner：页脚样板行 ----------

def test_copyright_line_removed():
    text = "第一条 用人单位应当依法建立用工制度\n中华人民共和国最高人民法院 版权所有\n第二条 本法自公布之日起施行"
    assert "版权所有" not in clean_text(text)


def test_icp_beian_line_removed():
    text = "第一条 为了完善劳动合同制度\n京ICP备05023036号\n第二条 本法自公布之日起施行"
    assert "京ICP备" not in clean_text(text)


def test_gongan_beian_number_line_removed():
    text = "第一条 为了完善劳动合同制度\n11040102700145号\n第二条 本法自公布之日起施行"
    assert "11040102700145" not in clean_text(text)


def test_hex_statistics_line_removed():
    text = "第一条 为了完善劳动合同制度\nf68e0d471f4a5e54fa5d62a9c18424c3\n第二条 本法自公布之日起施行"
    assert "f68e0d47" not in clean_text(text)


def test_breadcrumb_and_fontsize_prefix_removed():
    text = "所在位置：\n首页 > 法院资讯\n字号：\n第一条 为了完善劳动合同制度"
    cleaned = clean_text(text)
    assert "所在位置" not in cleaned
    assert "字号" not in cleaned
    # 面包屑正文行不是样板前缀，应保留（这里靠导航/短行规则另行处理）
    assert "第一条" in cleaned


def test_footer_contact_lines_removed():
    text = "第一条 为了完善劳动合同制度\n总机：67550114\n举报电话：67556131\n第二条 本法自公布之日起施行"
    cleaned = clean_text(text)
    assert "总机" not in cleaned
    assert "举报电话" not in cleaned


def test_structural_heading_not_removed_as_boilerplate():
    # 红线：结构标题白名单优先，样板规则不得吃掉章节/条文标题
    assert clean_text("第一章\n总则\n第五条") == "第一章\n总则\n第五条"


def test_normal_text_with_digit_not_removed():
    # 行内含长数字但不是"纯数字+号"形态，不误删
    text = "第一条 工资应当在20250801前支付完毕"
    assert "20250801" in clean_text(text)


# ---------- parser：TRS 正文区提取 ----------

def _write_html(tmp_path: Path, body: str) -> Path:
    html = f"<html><head><title>t</title></head><body>{body}</body></html>"
    path = tmp_path / "page.html"
    path.write_text(html, encoding="utf-8")
    return path


def test_zoom_container_preferred(tmp_path):
    body = (
        '<div class="crumb">所在位置：首页 > 资讯</div>'
        '<div class="txt_txt" id="zoom"><p>这是正文第一段，内容足够长，用来验证只取容器内文本。</p>'
        "<p>这是正文第二段，页脚不应出现在解析结果里。</p></div>"
        "<div class=\"footer\">版权所有 京ICP备05023036号</div>"
    )
    doc = parse_document(_write_html(tmp_path, body))
    assert "这是正文第一段" in doc.content
    assert "所在位置" not in doc.content
    assert "版权所有" not in doc.content


def test_no_zoom_falls_back_to_full_body(tmp_path):
    # 没有 zoom 容器：回退全页提取（行为与旧版一致），样板由 cleaner 兜底
    body = '<div><p>第一条 为了完善劳动合同制度</p><div>版权所有</div></div>'
    doc = parse_document(_write_html(tmp_path, body))
    assert "第一条" in doc.content
    assert "版权所有" not in doc.content  # cleaner 样板过滤兜底


def test_zoom_too_short_falls_back(tmp_path):
    # zoom 容器内容过短（<100 字符）视为误命中，回退全页
    body = (
        '<div id="zoom"><p>短</p></div>'
        "<p>第一条 为了完善劳动合同制度，本法自公布之日起施行，内容足够长作为真实正文使用，避免被短容器误判。</p>"
    )
    doc = parse_document(_write_html(tmp_path, body))
    assert "第一条" in doc.content
