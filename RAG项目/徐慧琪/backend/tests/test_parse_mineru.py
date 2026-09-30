# 产物定位必须递归——目录名随输入类型变化，写死路径是已踩过的坑
import pathlib
import subprocess

import pytest
from app.ingest.parse_mineru import locate_product_md, run_mineru


def test_locate_finds_pdf_layout(tmp_path):
    # pdf 走 txt/ 子目录
    d = tmp_path / "中华人民共和国民法典（上册）" / "txt"
    d.mkdir(parents=True)
    (d / "中华人民共和国民法典（上册）.md").write_text("内容", encoding="utf-8")
    assert locate_product_md(tmp_path).name == "中华人民共和国民法典（上册）.md"


def test_locate_finds_office_layout(tmp_path):
    # docx 走 office/ 子目录——与 pdf 不同，所以不能写死 txt/
    d = tmp_path / "中册.md" / "office"
    d.mkdir(parents=True)
    (d / "中册.md.md").write_text("内容", encoding="utf-8")
    assert locate_product_md(tmp_path).name == "中册.md.md"


def test_locate_picks_main_product_over_derived(tmp_path):
    # 派生产物带后缀、文件名更长；主产物是裸文件名
    d = tmp_path / "x" / "txt"
    d.mkdir(parents=True)
    (d / "x.md").write_text("主产物", encoding="utf-8")
    (d / "x_extra_derived.md").write_text("派生", encoding="utf-8")
    assert locate_product_md(tmp_path).name == "x.md"


def test_locate_raises_when_missing(tmp_path):
    # 找不到必须显式报错，不能返回 None 让调用方静默跳过整册
    with pytest.raises(FileNotFoundError):
        locate_product_md(tmp_path)


def test_run_mineru_passes_backend_flags_for_pdf(tmp_path, monkeypatch):
    # pdf 需要 -b/-m 控制解析方式
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_mineru(tmp_path / "a.pdf", tmp_path / "out")
    assert "-b" in captured["cmd"] and "pipeline" in captured["cmd"]


def test_run_mineru_omits_backend_flags_for_docx(tmp_path, monkeypatch):
    # docx 走 office 分支，不接受 -b/-m；传了会报错
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_mineru(tmp_path / "a.docx", tmp_path / "out")
    assert "-b" not in captured["cmd"] and "-m" not in captured["cmd"]


def test_run_mineru_raises_on_failure(tmp_path, monkeypatch):
    # 解析失败必须抛出，由编排层决定重试或中断——不能吞掉
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="MinerU 解析失败"):
        run_mineru(tmp_path / "a.pdf", tmp_path / "out")
