# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""测试集自身的完整性校验（不依赖模型，`not slow`）。

`data/test_dialogues.json` 是 Task9 三方对比评测的输入，也是 Task11 交付文档
「优化前后对比」的证据来源。数据错了，评测结论就是错的，所以这里先把
**数据集结构**钉死：

  · 11 类 coref_type 每类至少 1 条（含英文问句与对抗样本）；
  · ≥15 轮；
  · 每轮都声明 expected_resolved 与 expected_entity；
  · 非首轮不得标 `none`（`none` 只用于「完整问题、不该触发改写」的首轮）；
  · 对抗样本轮（无历史代词）用 `<unresolved>` 哨兵声明「**不得猜测**」，
    且该哨兵只允许出现在 `adversarial_no_candidate` 上。

这里**不**跑模型，也不断言系统当前能否达到 expected_resolved——系统的真实
表现由 evaluate_multiturn.py 实测记录，测试只保证「尺子本身是准的」。
"""
import json
import os

from src import config

REQUIRED_TYPES = {
    "none", "person_pronoun", "person_pronoun_en", "demonstrative",
    "demonstrative_gai", "demonstrative_en", "zero",
    "ellipsis_topic_switch", "ellipsis_what_about_en",
    "topic_back_reference", "adversarial_no_candidate",
}

UNRESOLVED = "<unresolved>"


def _load():
    with open(os.path.join(config.ROOT, "data", "test_dialogues.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


def test_dataset_covers_all_required_coref_types():
    data = _load()
    got = {t["coref_type"] for d in data["dialogues"] for t in d["turns"]}
    assert REQUIRED_TYPES <= got, f"缺少类型：{sorted(REQUIRED_TYPES - got)}"


def test_dataset_has_enough_turns():
    data = _load()
    n = sum(len(d["turns"]) for d in data["dialogues"])
    assert n >= 15, f"扩展测试集至少 15 轮，实际 {n}"


def test_every_turn_declares_expected_resolved():
    data = _load()
    missing = [(d["dialogue_id"], t["turn"]) for d in data["dialogues"]
               for t in d["turns"] if not t.get("expected_resolved")]
    assert not missing, f"缺 expected_resolved：{missing}"


def test_non_first_turns_have_coref_type():
    data = _load()
    bad = [(d["dialogue_id"], t["turn"]) for d in data["dialogues"]
           for t in d["turns"] if t["turn"] > 1 and t["coref_type"] == "none"]
    assert not bad, f"非首轮必须标注 coref_type（直通用 'passthrough'）：{bad}"


# ---------------------------------------------------------------------------
# 以下是本任务补充的加固校验（brief 之外，保护评测输入不被写坏）
# ---------------------------------------------------------------------------
def test_dialogue_ids_unique_and_turns_numbered():
    data = _load()
    ids = [d["dialogue_id"] for d in data["dialogues"]]
    assert len(ids) == len(set(ids)), f"dialogue_id 重复：{ids}"
    for d in data["dialogues"]:
        nums = [t["turn"] for t in d["turns"]]
        assert nums == list(range(1, len(nums) + 1)), \
            f"{d['dialogue_id']} 轮号不连续：{nums}"


def test_every_turn_declares_expected_entity():
    """每轮都要声明「改写后必须出现的公司全称」——指代消解准确率按它判定。"""
    data = _load()
    bad = [(d["dialogue_id"], t["turn"]) for d in data["dialogues"]
           for t in d["turns"]
           if not t.get("expected_entity") and t["coref_type"] != "adversarial_no_candidate"]
    assert not bad, f"缺 expected_entity：{bad}"


def test_expected_answer_facts_are_string_lists():
    data = _load()
    bad = []
    for d in data["dialogues"]:
        for t in d["turns"]:
            facts = t.get("expected_answer_facts")
            if facts is not None and (not isinstance(facts, list)
                                      or any(not isinstance(f, str) or not f
                                             for f in facts)):
                bad.append((d["dialogue_id"], t["turn"], facts))
    assert not bad, f"expected_answer_facts 必须是非空字符串列表：{bad}"


def test_unresolved_sentinel_only_on_adversarial_turns():
    data = _load()
    bad = [(d["dialogue_id"], t["turn"], t["coref_type"])
           for d in data["dialogues"] for t in d["turns"]
           if (t["expected_resolved"] == UNRESOLVED) !=
              (t["coref_type"] == "adversarial_no_candidate")]
    assert not bad, f"<unresolved> 哨兵只允许用于 adversarial_no_candidate：{bad}"
