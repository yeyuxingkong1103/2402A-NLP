"""文档标题站点后缀剥离测试（批次 3-1）。

背景：真实采集页 <title> 形如"工资支付暂行规定_人力资源和社会保障部"，
旧规则的后缀候选漏了部委名，导致下划线+站点名被原样存进 documents.title。
"""
from pathlib import Path

from app.ingest.title_normalizer import _extract_document_title


def _write_html(tmp_path: Path, name: str, title: str) -> Path:
    """写一个最小 HTML 页面，<title> 为指定文本。"""
    html = f"<html><head><title>{title}</title></head><body><p>正文内容。</p></body></html>"
    path = tmp_path / name
    path.write_text(html, encoding="utf-8")
    return path


def test_ministry_suffix_stripped(tmp_path: Path) -> None:
    """「工资支付暂行规定_人力资源和社会保障部」剥离后必须等于「工资支付暂行规定」。"""
    path = _write_html(tmp_path, "wage.html", "工资支付暂行规定_人力资源和社会保障部")
    title = _extract_document_title(path, "第一条 为保障劳动者取得劳动报酬的权利。")
    assert title == "工资支付暂行规定"


def test_ministry_short_form_suffix_stripped(tmp_path: Path) -> None:
    """简称部委站点（人社部等写法之一）也要剥离。"""
    path = _write_html(tmp_path, "wage2.html", "工资支付暂行规定_人力资源社会保障部")
    title = _extract_document_title(path, "第一条 为保障劳动者取得劳动报酬的权利。")
    assert title == "工资支付暂行规定"


def test_bureau_suffix_stripped(tmp_path: Path) -> None:
    """总局/局类站点后缀也要剥离。"""
    path = _write_html(tmp_path, "tax.html", "个人所得税专项附加扣除办法_国家税务总局")
    title = _extract_document_title(path, "第一条 根据个人所得税法制定本办法。")
    assert title == "个人所得税专项附加扣除办法"


def test_site_name_alone_still_blacklisted(tmp_path: Path) -> None:
    """<title> 整个就是部委站点名 → 视为提取失败，走人工核对标题。"""
    path = _write_html(tmp_path, "site.html", "人力资源和社会保障部")
    title = _extract_document_title(
        path, "第一条 内容。", explicit_title="工资支付暂行规定"
    )
    assert title == "工资支付暂行规定"


def test_normal_title_with_separator_kept(tmp_path: Path) -> None:
    """分隔符后的部分不是站点名 → 不得误剥（防止规则放太宽）。"""
    path = _write_html(tmp_path, "normal.html", "企业职工培训规定")
    title = _extract_document_title(path, "第一条 内容。")
    assert title == "企业职工培训规定"
