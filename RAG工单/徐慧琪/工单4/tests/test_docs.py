# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

REQUIRED = [
    "图像解析与检索专项说明.md",
    "答案定位报告.md",
    "技术文档.md",
    "用户手册.md",
    "演示视频录制脚本.md",
]

PLACEHOLDER = re.compile(r"(TBD|TODO|待补充|待填写|XXX|FIXME)")


@pytest.mark.parametrize("name", REQUIRED)
def test_doc_exists_and_not_empty(name):
    p = DOCS / name
    assert p.exists(), f"缺文档：{name}"
    assert len(p.read_text(encoding="utf-8")) > 800, f"{name} 内容过少"


@pytest.mark.parametrize("name", REQUIRED)
def test_doc_has_no_placeholders(name):
    p = DOCS / name
    if not p.exists():
        pytest.skip("文档不存在")
    txt = p.read_text(encoding="utf-8")
    hits = PLACEHOLDER.findall(txt)
    assert not hits, f"{name} 残留占位符：{set(hits)}"


def test_special_doc_covers_hard_requirements():
    """专项说明必须覆盖工单备注的两条硬性要求。"""
    txt = (DOCS / "图像解析与检索专项说明.md").read_text(encoding="utf-8")
    assert "CLIP" in txt
    assert "多模态" in txt
    assert "图区" in txt or "图区检测" in txt
    assert "max_tokens" in txt or "2000" in txt


def test_location_report_covers_16_questions():
    p = DOCS / "答案定位报告.md"
    if not p.exists():
        pytest.skip("未生成")
    txt = p.read_text(encoding="utf-8")
    for qid in (5, 6, 1, 2, 3, 4, 260, 95, 33, 34, 957, 793, 795, 543, 531, 207):
        assert f"id {qid}" in txt or f"id{qid}" in txt, f"缺 id {qid}"


def test_tech_doc_contains_work_order_id():
    txt = (DOCS / "技术文档.md").read_text(encoding="utf-8")
    assert "人工智能NLP-RAG-图像内容解析及检索优化" in txt


def test_recording_script_has_timeline():
    txt = (DOCS / "演示视频录制脚本.md").read_text(encoding="utf-8")
    assert "分镜" in txt or "时间" in txt
    assert "id 5" in txt or "id5" in txt


# ============================================================================
# 复核新增（review round 1, Minor #9）：数字守卫
#
# 上面的 brief 测试只查「存在/长度/关键词」，一份把数字删光或写错数字的文档
# 照样能过。以下断言把**今天已经真实跑出的关键数字**钉在文档里，并守住三条
# 曾经的失真点：旧报告数字（0.1875）、未实现的限流、压测参数口径。
# ============================================================================

_SUPERSEDED = "0.1875"          # Task 17 的旧 answer_accuracy，已被 0.875 取代


def test_index_composition_is_stated_and_adds_up():
    """三库构成是全文最重要的规模数字：既要在，也要加得上。"""
    assert 4436 + 681 + 162 == 5279, "文档口径：text+table+image = 总块数"
    sp = (DOCS / "图像解析与检索专项说明.md").read_text(encoding="utf-8")
    for token in ("4436", "681", "162", "5279"):
        assert token in sp, f"专项说明缺索引构成数字 {token}"
    for name in ("技术文档.md", "用户手册.md", "演示视频录制脚本.md"):
        assert "5279" in (DOCS / name).read_text(encoding="utf-8"), f"{name} 缺总块数 5279"


def test_docs_agree_on_headline_metrics():
    """同一指标在多份文档里出现时，数字必须一致（当前口径：eval_full_04.json，
    `ollama_url` 的 IPv4 修复后那次运行；p50 3888.6 是此前的 RC-2 口径，已作废）。"""
    for name in ("图像解析与检索专项说明.md", "技术文档.md", "演示视频录制脚本.md",
                 "答案定位报告.md"):
        txt = (DOCS / name).read_text(encoding="utf-8")
        assert "0.875" in txt, f"{name} 缺 answer_accuracy 实测值"
        assert "1730.3" in txt, f"{name} 缺 p50 实测值（当前口径 1730.3 ms）"


def test_special_doc_states_retrieval_metrics_not_just_accuracy():
    sp = (DOCS / "图像解析与检索专项说明.md").read_text(encoding="utf-8")
    for token in ("0.9375", "0.7396", "0.6979", "0.6422"):
        assert token in sp, f"专项说明缺检索指标 {token}"


def test_both_shortfalls_are_stated_plainly():
    """未达标项必须白纸黑字写出，不得只报成绩；也不得把「热态达标」写成笼统「已达标」。
    时延结论自 2026-10-10 复核起为分场景口径：热态达标，冷启动 41.7 s、
    高并发（100 档服务崩溃）未达标。"""
    sp = (DOCS / "图像解析与检索专项说明.md").read_text(encoding="utf-8")
    assert "答案准确率 ≥ 90%：未达成" in sp, "必须明写准确率未达成"
    assert "冷启动与高并发未达标" in sp, "3 秒项的未达标部分必须明写（冷启动/高并发）"
    assert "100 档服务进程崩溃" in sp, "高并发 100 档崩溃不得隐去"
    tech = (DOCS / "技术文档.md").read_text(encoding="utf-8")
    assert "**未达成**" in tech, "技术文档的准确率未达成结论不得被删改"
    assert "**高并发稳定性未达标**" in tech, "技术文档必须明写高并发未达标"
    assert "冷启动与并发档未达标" in tech, "技术文档的时延分项结论不得被删改"


def test_image_answers_keep_measured_values():
    """id 5/id 6 的关键答案值：删数字（或改成编造值）即失败。"""
    sp = (DOCS / "图像解析与检索专项说明.md").read_text(encoding="utf-8")
    for token in ("14.0%", "-2.0%", "10/10", "珠海销售处", "武汉销售处"):
        assert token in sp, f"专项说明缺图像题实测要点 {token}"


def test_image_traceability_is_pinned_to_image_path():
    """图像溯源口径：按引用项的 image_path 选取，且能落到具体原始图文件。"""
    for name in ("图像解析与检索专项说明.md", "用户手册.md"):
        txt = (DOCS / name).read_text(encoding="utf-8")
        assert "figuretext#39#fig4" in txt, f"{name} 缺 id5 引用来源ID"
        assert "招股说明书2_p39_7fb46462e43d.png" in txt, f"{name} 缺 id5 原始图文件名"
        assert "image_path" in txt, f"{name} 必须说明按 image_path 选取"
        assert "含图像块时" not in txt, (
            f"{name} 仍有「引用里含图像块时」的旧写法：按 block_type 过滤会让"
            "figuretext 描述块（block_type=text）丢失原图"
        )


def test_no_doc_presents_the_superseded_report_numbers():
    """Task 17 的旧数字不得当作成绩出现；只允许录像脚本作为「别打开旧报告」的警告。"""
    for name in ("图像解析与检索专项说明.md", "答案定位报告.md",
                 "技术文档.md", "用户手册.md"):
        assert _SUPERSEDED not in (DOCS / name).read_text(encoding="utf-8"), (
            f"{name} 出现旧报告数字 {_SUPERSEDED}"
        )
    lines = (DOCS / "演示视频录制脚本.md").read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if _SUPERSEDED in line:
            # 旧数字只允许出现在「这是 Task 17 的旧报告、不要打开」的警告里
            window = "".join(lines[max(0, i - 1):i + 2])
            assert "旧" in window, f"演示视频录制脚本.md:{i + 1} 旧数字只能作为警告出现"


def test_recording_script_points_at_current_report():
    """录制时必须打开与解说数字一致的那一份（当前口径 full_04.md / eval_full_04.json），
    旧口径 rc2 只能作为对照出现，避免照着旧数字口播。"""
    script = (DOCS / "演示视频录制脚本.md").read_text(encoding="utf-8")
    assert "eval_full_04.json" in script
    assert "检索精确度报告_full_04.md" in script, "录制时必须打开当前口径的那一份"
    assert "检索精确度报告_full_04_rc2.md" in script, "旧口径需作为对照出现，避免念错数字"
    assert "1730.3" in script, "脚本里的 p50 必须与当前口径一致"


_NEGATED_THROTTLE = ("未接线", "尚未接线", "没有实现", "没有请求限流", "无请求限流",
                     "不要声称")


def test_unimplemented_throttling_is_never_claimed():
    """`llm_max_concurrency` 只是预留配置项：任何「限流」表述都必须带否定语。"""
    for name in ("技术文档.md", "演示视频录制脚本.md"):
        for i, line in enumerate((DOCS / name).read_text(encoding="utf-8").splitlines(), 1):
            if "限流" in line:
                assert any(neg in line for neg in _NEGATED_THROTTLE), (
                    f"{name}:{i} 声称了代码里不存在的限流：{line.strip()[:80]}"
                )


def test_tech_doc_documents_real_loadtest_defaults():
    """压测每档请求数的真实口径：缺省 3×最大并发；小值会被上调，不是照跑 60。"""
    tech = (DOCS / "技术文档.md").read_text(encoding="utf-8")
    assert "3 × 最大并发" in tech
    assert "每档 100 请求" in tech
    assert "每档 60 请求" not in tech, "旧的错误示例（每档 60 请求）不得残留"
