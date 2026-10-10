# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""Streamlit 界面单测：只覆盖可纯函数化的部分（文案、引用格式化、时延如实呈现）。

界面本身（浏览器交互、上传建库、原图展示）依赖真实索引与模型，只能在索引空闲时
人工验证；本文件绝不导入 streamlit、绝不构造 RAGPipeline（会锁 data/qdrant）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag04.config import latency_verdict
from rag04.ui.app import (
    UI_TEXT,
    build_uploaded,
    format_answer_lang,
    format_latency,
    image_citations,
    needs_overwrite_confirm,
    render_citations,
    safe_upload_name,
    t,
    warm_up,
)

# 采集期快照：导入 app 模块后 streamlit 仍不得进 sys.modules，
# 证明 `import streamlit` 只发生在 main() 内（单测不拖入整个 Streamlit 运行时）。
_STREAMLIT_IMPORTED_AT_IMPORT = "streamlit" in sys.modules

ROOT = Path(__file__).resolve().parents[1]


def test_ui_text_has_both_languages():
    assert "zh" in UI_TEXT and "en" in UI_TEXT
    assert UI_TEXT["zh"].keys() == UI_TEXT["en"].keys()


def test_t_returns_localized_string():
    assert t("ask_button", "zh") == "提问"
    assert t("ask_button", "en") == "Ask"


def test_t_falls_back_to_key_for_unknown():
    assert t("nonexistent_key", "zh") == "nonexistent_key"


def test_t_english_question_hint_exists():
    """界面必须支持中英文输入。"""
    assert "中文" in UI_TEXT["zh"]["question_label"] or "Chinese" in UI_TEXT["zh"]["question_label"]
    assert "English" in UI_TEXT["en"]["question_label"] or "英文" in UI_TEXT["en"]["question_label"]


def test_render_citations_formats_page_and_type():
    """断言整行：只查 "38"/"image" 会被 source_id 里的同样子串蒙混过关（复核 #4）。"""
    lines = render_citations([
        {"page": 38, "block_type": "image", "source_id": "image#38#fig0", "score": 0.91},
    ])
    assert lines == ["p38 | image | image#38#fig0 | score=0.91"]


def test_render_citations_one_line_per_citation():
    lines = render_citations([
        {"page": 1, "block_type": "text", "source_id": "text#1#0", "score": 0.5},
        {"page": 2, "block_type": "table", "source_id": "table#2#0", "score": 0.4},
    ])
    assert len(lines) == 2 and lines[0].startswith("p1 | text") and lines[1].startswith("p2 | table")


def test_render_citations_tolerates_missing_keys():
    """字段缺失必须走 .get 兜底而不是抛 KeyError：界面不因引用字典不全而崩。"""
    assert render_citations([{}]) == ["pNone | None | None | score=None"]
    assert render_citations([{"page": 3, "block_type": "table"}]) == \
        ["p3 | table | None | score=None"]


def test_render_citations_empty():
    assert render_citations([]) == []


def test_ui_text_covers_required_features():
    for key in ("title", "upload_label", "ask_button", "answer_label",
                "citations_label", "metrics_label", "image_label", "health_label"):
        assert key in UI_TEXT["zh"], f"缺文案：{key}"
        assert key in UI_TEXT["en"], f"缺英文文案：{key}"


# --- 补充 1：中英双语提问与答案语言标注 ---


def test_ui_text_exposes_answer_language_labels():
    """中英双语界面：必须能标注答案回来的是中文还是英文。"""
    for key in ("answer_lang", "lang_name_zh", "lang_name_en"):
        assert key in UI_TEXT["zh"], f"缺文案：{key}"
        assert key in UI_TEXT["en"], f"缺英文文案：{key}"


def test_format_answer_lang_is_localized():
    assert format_answer_lang("zh", "zh") == "中文"
    assert format_answer_lang("en", "zh") == "英文"
    assert format_answer_lang("zh", "en") == "Chinese"
    assert format_answer_lang("en", "en") == "English"


def test_format_answer_lang_echoes_unknown_value():
    """未知/空语言值原样回显，不臆造也不隐藏。"""
    assert format_answer_lang("mixed", "zh") == "mixed"
    assert format_answer_lang("", "zh") == ""


def test_format_answer_lang_handles_none():
    """复核 #8：注解写 str 就必须真返回 str，None 不得漏出去。"""
    assert format_answer_lang(None, "zh") == ""
    assert isinstance(format_answer_lang(None, "en"), str)


def test_metrics_label_names_citation_count_not_precision():
    """复核 #1：界面拿不到金标算不出 precision，标签必须声明它是「条数」。"""
    assert "精确度" not in UI_TEXT["zh"]["metrics_label"]
    assert "precision" not in UI_TEXT["en"]["metrics_label"].lower()
    assert UI_TEXT["zh"]["metrics_label"] and UI_TEXT["en"]["metrics_label"]


def test_ui_text_has_no_dead_lang_label():
    """复核 #5：语言选择框用固定的双语标签，lang_label 是死键，已删除。"""
    assert "lang_label" not in UI_TEXT["zh"] and "lang_label" not in UI_TEXT["en"]


def test_upload_path_keys_are_wired():
    """复核 #5：no_pdf 必须真的被界面引用（首跑最常见失败：未建库）。"""
    import inspect

    import rag04.ui.app as app_mod

    main_src = inspect.getsource(app_mod.main)
    assert 't("no_pdf", lang)' in main_src
    assert UI_TEXT["zh"]["no_pdf"] and UI_TEXT["en"]["no_pdf"]


# --- 补充 2：时延与 3 秒预算如实呈现（实测 p50 ≈ 3.9s，超预算必须说超）---


def test_format_latency_reports_over_budget_honestly():
    """超预算时展示实测原值 + 预算值，并明说「超出」，不截断不美化。"""
    s = format_latency(3888.6, 3000.0, "zh")
    assert "3888.6" in s        # 实测 p50 档位原样出现（一位小数）
    assert "3000.0" in s
    assert UI_TEXT["zh"]["latency_over"].format(ms=3888.6, budget=3000.0) == s


def test_format_latency_reports_within_budget():
    s = format_latency(1234.6, 3000.0, "zh")
    assert "1234.6" in s        # 仅显示保留一位小数，判定仍用原值
    assert UI_TEXT["zh"]["latency_within"].format(ms=1234.6, budget=3000.0) == s


def test_format_latency_just_over_budget_is_not_self_contradictory():
    """复核 #2：3000.4 ms 超预算，不得渲染成「3000 ms 超出 3000 ms 预算」。"""
    s = format_latency(3000.4, 3000.0, "zh")
    assert "3000.4" in s and "超出" in s
    assert "3000 ms" not in s
    assert "3000.4" in format_latency(3000.4, 3000.0, "en")


def test_format_latency_boundary_equals_budget_counts_as_within():
    """与 rag04.api.server 判定口径一致：latency_ms == budget 视为达标。"""
    assert format_latency(3000.0, 3000.0, "zh") == \
        UI_TEXT["zh"]["latency_within"].format(ms=3000.0, budget=3000.0)


def test_format_latency_is_localized():
    assert "over" in format_latency(3900.0, 3000.0, "en")
    assert "within" in format_latency(100.0, 3000.0, "en")


# --- 终审修正B：判定唯一实现 + 上传真入库 + 预热 ---


def test_format_latency_uses_the_shared_verdict_helper():
    """界面显示/判定必须由 ``rag04.config.latency_verdict`` 决定（接口/压测同一函数）。

    3000.04 ms 旧实现按原值判「超出」，却渲染成「3000.0 ms，超出 3000.0 ms 预算」。
    """
    shown, within = latency_verdict(3000.04, 3000.0)
    assert (shown, within) == (3000.0, True)
    s = format_latency(3000.04, 3000.0, "zh")
    assert s == UI_TEXT["zh"]["latency_within"].format(ms=3000.0, budget=3000.0)
    assert "超出" not in s and "3000.04" not in s


def test_format_latency_agrees_with_helper_on_many_values():
    """逐值对账：显示里的数字与「在/超出预算」结论都必须等于共享判定的输出。"""
    for ms, budget in [(2999.96, 3000.0), (3000.0, 3000.0), (3000.06, 3000.0),
                       (3888.6, 3000.0), (100.0, 3000.0), (13385.0, 3000.0)]:
        shown, within = latency_verdict(ms, budget)
        s = format_latency(ms, budget, "zh")
        assert f"{shown:.1f}" in s, (ms, s)
        assert ("超出" in s) is not within, (ms, s)


def test_format_latency_displayed_number_matches_verdict_side():
    """显示的毫秒值必须与结论同侧：不能在「预算内」分支里显示取整后的超预算值。"""
    for ms in (3000.04, 3000.06, 3000.4, 2999.99):
        shown, within = latency_verdict(ms, 3000.0)
        expected = UI_TEXT["zh"]["latency_within" if within
                            else "latency_over"].format(ms=shown, budget=3000.0)
        assert format_latency(ms, 3000.0, "zh") == expected


CORPUS = ("招股说明书1.pdf", "招股说明书2.pdf")


class _FakePipe:
    """build() 替身：记录调用参数 + 携带 corpus，绝不碰索引/模型。"""

    def __init__(self, error: Exception | None = None, corpus=CORPUS) -> None:
        self.calls: list[dict] = []
        self.error = error
        self.s = SimpleNamespace(corpus=corpus)

    def build(self, names=None, reset=True):
        self.calls.append({"names": names, "reset": reset})
        if self.error is not None:
            raise self.error
        return []


def test_build_uploaded_appends_a_new_document_without_resetting(tmp_path):
    """新文档（不在语料里）→ 追加：names = 语料 + 新文件、reset=False。

    这守住「上传不毁库」：旧修正版曾用 reset=True，上传任意 PDF 会把两份语料
    从索引里抹掉，工单 16 题当场答不了。新文档带新 chunk_id，无需清库。
    """
    dest = tmp_path / "新文档.pdf"
    pipe = _FakePipe()
    build_uploaded(pipe, dest, b"%PDF-1.4 fake", "新文档.pdf")
    assert dest.read_bytes() == b"%PDF-1.4 fake"
    assert pipe.calls == [{"names": list(CORPUS) + ["新文档.pdf"],
                           "reset": False}]
    assert pipe.s.corpus == CORPUS          # 配置未被就地改写（tuple 仍是两份语料）


def test_build_uploaded_rebuilds_when_a_corpus_file_is_overwritten(tmp_path):
    """覆盖语料同名文件 → 正文变了、chunk_id 变了：必须 reset 清库后按语料全量重建。"""
    dest = tmp_path / "招股说明书1.pdf"
    pipe = _FakePipe()
    build_uploaded(pipe, dest, b"%PDF-1.4 v2", "招股说明书1.pdf")
    assert dest.read_bytes() == b"%PDF-1.4 v2"
    assert pipe.calls == [{"names": list(CORPUS), "reset": True}]


def test_build_uploaded_propagates_failure_after_writing(tmp_path):
    """建库失败原样抛出（界面本地化提示 + 原始错误），绝不吞成「成功」。"""
    dest = tmp_path / "x.pdf"
    pipe = _FakePipe(error=RuntimeError("Index already accessed"))
    with pytest.raises(RuntimeError, match="already accessed"):
        build_uploaded(pipe, dest, b"%PDF", "x.pdf")
    assert dest.exists()                       # 文件已落盘，缺的是索引重建


def test_needs_overwrite_confirm_for_corpus_and_existing_files(tmp_path):
    corpus = ("招股说明书1.pdf", "招股说明书2.pdf")
    existing = tmp_path / "旧上传.pdf"
    existing.write_bytes(b"%PDF")
    assert needs_overwrite_confirm("招股说明书1.pdf", corpus,
                                   tmp_path / "招股说明书1.pdf") is True
    # 语料文件即使不在磁盘上也必须确认（源 PDF 未纳入版本管理，无法恢复）
    assert needs_overwrite_confirm("招股说明书1.pdf", corpus,
                                   tmp_path / "不存在.pdf") is True
    assert needs_overwrite_confirm("旧上传.pdf", corpus, existing) is True
    assert needs_overwrite_confirm("全新.pdf", corpus,
                                   tmp_path / "全新.pdf") is False


def test_main_gates_overwrite_behind_a_confirmation():
    """破坏性写入必须过闸：同名文件要勾选确认，未勾选不得落盘/建库。"""
    import inspect

    import rag04.ui.app as app_mod

    main_src = inspect.getsource(app_mod.main)
    assert "needs_overwrite_confirm(" in main_src
    assert 't("overwrite_confirm", lang)' in main_src
    assert 't("overwrite_required", lang)' in main_src
    assert "build_uploaded(" in main_src
    # 语料同名 vs 普通已存在文件：警告与后续语义必须分开（一个重建、一个追加）
    assert "fname in settings.corpus" in main_src
    assert '"overwrite_warn_corpus" if replacing' in main_src


def test_ui_text_distinguishes_append_from_corpus_rebuild():
    """追加（新文档）与按语料重建（覆盖语料）必须有各自的双语文案。"""
    for key in ("indexing", "indexing_rebuild", "upload_ok", "upload_ok_rebuild",
                "overwrite_warn", "overwrite_warn_corpus"):
        assert UI_TEXT["zh"][key] and UI_TEXT["en"][key], f"缺双语文案：{key}"
    assert UI_TEXT["zh"]["indexing_rebuild"] != UI_TEXT["zh"]["indexing"]
    assert UI_TEXT["zh"]["upload_ok_rebuild"] != UI_TEXT["zh"]["upload_ok"]
    assert "追加" in UI_TEXT["zh"]["upload_ok"]
    assert "重建" in UI_TEXT["zh"]["upload_ok_rebuild"]
    # 语义纠偏后不得再声称「索引里只剩上传的文件」
    for lang in ("zh", "en"):
        for key, text in UI_TEXT[lang].items():
            assert "只包含该文件" not in text and "only that document" not in text, \
                f"{lang}.{key} 仍声称上传会清空索引：{text}"


def test_main_reports_build_failure_in_localized_form():
    """重建失败给本地化文案 + 原始错误，不把 Streamlit 裸栈甩给用户。"""
    import inspect

    import rag04.ui.app as app_mod

    main_src = inspect.getsource(app_mod.main)
    assert 't("upload_error", lang).format(' in main_src
    assert "build_uploaded(" in main_src
    try_block = main_src.split("build_uploaded(")[0].rsplit("try:", 1)
    assert len(try_block) == 2, "build_uploaded 必须在 try 内调用"


def test_main_has_no_hardcoded_english_build_strings():
    """复核：'Indexing...' / 'OK' 是硬编码英文，必须走 UI_TEXT 双语。"""
    import inspect

    import rag04.ui.app as app_mod

    main_src = inspect.getsource(app_mod.main)
    assert 'st.spinner("Indexing...")' not in main_src
    assert 'st.success("OK")' not in main_src
    # 两条分支各自的 spinner / success 文案都从 UI_TEXT 取
    assert 't("indexing_rebuild" if replacing else "indexing",' in main_src
    assert 't("upload_ok_rebuild" if replacing' in main_src
    for key in ("indexing", "indexing_rebuild", "upload_ok", "upload_ok_rebuild",
                "upload_error", "upload_bad_name", "overwrite_warn",
                "overwrite_warn_corpus", "overwrite_confirm", "overwrite_required",
                "warmup_ok", "warmup_warn"):
        assert UI_TEXT["zh"][key] and UI_TEXT["en"][key], f"缺双语文案：{key}"


class _FakeWarmPipe:
    def __init__(self, error: Exception | None = None) -> None:
        self.warmed = 0
        self.error = error

    def warmup(self):
        self.warmed += 1
        if self.error is not None:
            raise self.error


def test_warm_up_preloads_and_reports_success():
    """界面进程必须预热（服务端启动就预热，界面此前没有，首答多付约 8 秒）。"""
    pipe = _FakeWarmPipe()
    assert warm_up(pipe) is None
    assert pipe.warmed == 1


def test_warm_up_is_failure_tolerant_and_returns_reason():
    """预热失败只记录并返回原因：界面照常可用，首答变慢有据可查。"""
    pipe = _FakeWarmPipe(error=RuntimeError("reranker 缺失"))
    msg = warm_up(pipe)
    assert msg is not None and "reranker 缺失" in msg
    assert pipe.warmed == 1


def test_main_warms_the_pipeline_and_surfaces_state():
    """get_pipe 里要调 warm_up，并把结果落到本地化 caption（失败原因对用户可见）。"""
    import inspect

    import rag04.ui.app as app_mod

    main_src = inspect.getsource(app_mod.main)
    assert "warm_up(p)" in main_src
    assert 't("warmup_ok", lang)' in main_src
    assert 't("warmup_warn", lang).format(' in main_src


def test_format_latency_does_not_clamp_value():
    """极端超预算值也必须原样呈现（防止有人「取整到预算内」）。"""
    assert "13385.0" in format_latency(13385.0, 3000.0, "zh")   # 实测 p95 档位


# --- 补充 4：上传文件名消毒（复核 #6）---


def test_safe_upload_name_keeps_plain_pdf():
    assert safe_upload_name("招股说明书1.pdf") == "招股说明书1.pdf"
    assert safe_upload_name("PROSPECTUS.PDF") == "PROSPECTUS.PDF"


def test_safe_upload_name_strips_path_components():
    """type=["pdf"] 只是前端过滤：路径成分必须剥掉，否则可穿越项目根目录。"""
    assert safe_upload_name("..\\..\\x.pdf") == "x.pdf"
    assert safe_upload_name("../../x.pdf") == "x.pdf"
    assert safe_upload_name("C:\\tmp\\x.pdf") == "x.pdf"
    assert safe_upload_name("sub/dir/x.pdf") == "x.pdf"


def test_safe_upload_name_rejects_non_pdf():
    assert safe_upload_name("evil.exe") is None
    assert safe_upload_name("..\\..\\x.pdf.exe") is None
    assert safe_upload_name("x.pdf/..\\y") is None      # 末段不是 .pdf
    assert safe_upload_name("") is None
    assert safe_upload_name(None) is None


# --- 补充 5：图像溯源按 image_path 选取（RC-2 后图题头条引用是图描述文本块）---

_FIGTEXT_39 = {"page": 39, "block_type": "text", "source_id": "figuretext#39#fig4",
               "image_path": "D:/data/figures/招股说明书2_p39_7fb46462e43d.png",
               "score": 0.91}


def test_image_citations_includes_figure_description_text_block():
    """id5/id6 的真实形态：block_type=text + image_path，必须能带出原图。"""
    assert image_citations([_FIGTEXT_39]) == [_FIGTEXT_39]


def test_image_citations_keeps_image_block_behaviour():
    """回归：block_type=image 且带路径的引用不能被弄丢。"""
    c = {"page": 153, "block_type": "image", "source_id": "image#153#fig38",
         "image_path": "D:/data/figures/招股说明书1_p153_abc.png", "score": 0.88}
    assert image_citations([c]) == [c]


def test_image_citations_excludes_citation_without_image_path():
    assert image_citations([
        {"page": 10, "block_type": "text", "source_id": "text#10#0", "image_path": ""},
        {"page": 12, "block_type": "table", "source_id": "table#12#0", "image_path": ""},
    ]) == []


def test_image_citations_tolerates_missing_or_none_image_path():
    """缺键 / None 都不得抛错（引用字典来自检索层，字段不保证齐全）。"""
    assert image_citations([
        {"page": 10, "block_type": "text", "source_id": "text#10#0"},
        {"page": 11, "block_type": "image", "source_id": "image#11#fig0",
         "image_path": None},
    ]) == []
    assert image_citations([]) == []


def test_image_citations_dedupes_by_path_and_keeps_rank_order():
    """id793 形态：同一张图被 image 块与描述块同时引用 —— 去重且保留靠前者。"""
    top = {"page": 153, "block_type": "image", "source_id": "image#153#fig38",
           "image_path": "p153.png", "score": 0.9}
    dup = {"page": 1, "block_type": "text", "source_id": "figuretext#1#fig0",
           "image_path": "p153.png", "score": 0.7}
    other = {"page": 310, "block_type": "text", "source_id": "figuretext#310#fig88",
             "image_path": "p310.png", "score": 0.5}
    assert image_citations([top, dup, other]) == [top, other]


def test_main_uses_image_citations_helper():
    """守住真实缺陷点：main() 必须走 image_citations，而不是旧的 block_type 过滤。"""
    import inspect

    import rag04.ui.app as app_mod

    main_src = inspect.getsource(app_mod.main)
    assert "image_citations(ans.citations)" in main_src
    assert '"image"' not in main_src or 'block_type") == "image"' not in main_src


# --- 补充 3：导入期安全 ---


def test_app_module_does_not_import_streamlit_at_import_time():
    """streamlit 只在 main() 内导入：否则单测会被整套运行时拖慢。"""
    assert _STREAMLIT_IMPORTED_AT_IMPORT is False


def test_module_executes_as_a_script_without_src_on_sys_path():
    """``streamlit run src/rag04/ui/app.py`` 是**脚本**执行：模块级不得 import rag04。

    这条路径下 sys.path 里没有 ``src``（路径兜底在 main() 里才补上），任何模块级
    ``import rag04.*`` 都会让界面启动即 ``ModuleNotFoundError``。用子进程复现
    该执行方式（不导入 streamlit：``run_path`` 的 ``__name__`` 不是 ``__main__``）。
    """
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    code = ("import runpy; d = runpy.run_path(r'src/rag04/ui/app.py'); "
            "assert callable(d['format_latency']) and callable(d['build_uploaded'])")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
        check=False,          # 出错不抛，改为断言 returncode（stderr 更好读）
    )
    assert proc.returncode == 0, proc.stderr


def _load_run_ui():
    """加载 scripts/run_ui.py 而不执行 __main__ 分支（不拉起子进程）。"""
    spec = importlib.util.spec_from_file_location(
        "rag04_run_ui_under_test", ROOT / "scripts" / "run_ui.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_ui_import_is_side_effect_free_and_targets_app():
    mod = _load_run_ui()
    app = mod.ROOT / "src" / "rag04" / "ui" / "app.py"
    assert app.is_file(), f"启动脚本指向的界面文件不存在：{app}"
    assert callable(mod.main)
