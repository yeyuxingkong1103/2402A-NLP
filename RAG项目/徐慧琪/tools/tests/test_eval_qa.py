# 指标算法是纯函数，用构造数据测；报告渲染只验"该有的都有"，
# 真跑由 Task 13 的冒烟覆盖。每个指标都要有一条"应当为假/为 0"的用例——
# 恒真的指标函数是最典型的"不可能失败的测试"。
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from eval_qa import compute_metrics, recall_at, render_report, topk_hit
from latency_report import LATENCY_HISTORY_END, LATENCY_HISTORY_HEADER, LATENCY_HISTORY_START

BLOCKS = [{"article_no": 584}, {"article_no": 577}, {"article_no": 1}]


def test_recall_at_finds_gold_within_k():
    assert recall_at(BLOCKS, [577], 3) is True
    # k=2 时 gold 恰落在**最后一个被纳入的位置**：只测"k 之外被排除"的话，
    # 切片被悄悄缩成 [:k-1]（Recall@20/50 一起缩水）也照样全绿。
    # 与下面那条"k 之外为假"合起来，边界两个方向才都被钉住
    assert recall_at(BLOCKS, [577], 2) is True


def test_recall_at_misses_gold_beyond_k():
    assert recall_at(BLOCKS, [577], 1) is False


def test_recall_at_requires_all_gold_articles():
    # 多个标准条款时，缺一不可——部分召回在"这条到底答没答对"上是模糊的
    assert recall_at(BLOCKS, [584, 999], 3) is False


def test_topk_hit_uses_prefix_only():
    assert topk_hit(BLOCKS, [577], 3) is True
    assert topk_hit(BLOCKS, [577], 2) is True
    assert topk_hit(BLOCKS, [1], 2) is False


def _row(**kwargs):
    base = {"recalled_20": [584], "recalled_50": [584], "top3": [584],
            "gold": [584], "status": "ok", "latency": 1.0,
            "cite_ok": True, "exact_top1": True, "category": "条款号直问"}
    base.update(kwargs)
    return base


def test_compute_metrics_counts_each_rate():
    rows = [_row(), _row(status="abstain", cite_ok=False),
            _row(gold=[999], recalled_20=[1], recalled_50=[1], top3=[1],
                 cite_ok=True, exact_top1=False)]
    metrics = compute_metrics(rows)
    assert metrics["n"] == 3
    assert metrics["recall20"] == 2 / 3
    # recall50 / cite_rate / error_rate 也各钉一个数：没有断言的指标，
    # 实现退化成恒真也无人报错，等于白算
    assert metrics["recall50"] == 2 / 3
    assert metrics["cite_rate"] == 2 / 3
    assert metrics["error_rate"] == 0.0
    assert metrics["abstain_rate"] == 1 / 3
    assert metrics["top3_rate"] == 2 / 3


def test_compute_metrics_top3_rate_pins_the_k3_call_site():
    # AC-4 的 85% 门挂在这个 k=3 调用点上，此前没有任何测试：忠实变异把它改成
    # k=5 全套件仍绿（top3 字段存的是精排全量 5 条，没人用「gold 在 k 之外」的
    # 行打过它，而 brief 的注释还教人"改看 top5 不必重跑"，手滑改大会静默放松门）。
    # 两行合一，0.5 同时钉住两侧——k≤2 或 k≥4 都算不出 0.5，只有 k=3 对
    rows = [_row(top3=[1, 2, 3, 584]), _row(top3=[1, 2, 584, 3])]
    assert compute_metrics(rows)["top3_rate"] == 0.5


def test_compute_metrics_reports_zero_not_error_on_empty():
    # 空输入返回 0 值而不是抛 ZeroDivisionError，否则一次空跑会毁掉整轮
    metrics = compute_metrics([])
    assert metrics["n"] == 0
    assert metrics["recall20"] == 0.0


def test_compute_metrics_separates_direct_article_subset():
    # AC-2 只在"条款号直问"子集上算，混进语义题会把它的门槛稀释掉。
    # 子集里刻意放一条未命中的直问行、语义行则改成命中：分子若退化成
    # "直问题数"、或把语义题也算进来，全命中式的 fixture 都发现不了，
    # 只有 0.5 这个真实值能把它们打出来
    rows = [_row(exact_top1=True), _row(exact_top1=False),
            _row(category="语义改写", exact_top1=True)]
    metrics = compute_metrics(rows)
    assert metrics["direct_n"] == 2
    assert metrics["direct_rate"] == 0.5


def test_compute_metrics_error_rate_counts_failures():
    # error 与 abstain 绝不能混（answer.py：把服务故障答成"查无此条"是最危险的
    # 失败方向）。只断言 "error_rate == 0" 的用例杀不掉硬编码 0.0 的实现——
    # 恒零是恒真的孪生形态，同样要杀，所以这里必须有真的 error 行，
    # 并同时钉住它没被算进降级率
    rows = [_row(), _row(status="error", cite_ok=False)]
    metrics = compute_metrics(rows)
    assert metrics["error_rate"] == 0.5
    assert metrics["abstain_rate"] == 0.0


def test_compute_metrics_latency_percentiles():
    rows = [_row(latency=float(i)) for i in range(1, 101)]
    metrics = compute_metrics(rows)
    assert metrics["p50"] == 50.0
    assert metrics["p95"] == 95.0


def test_compute_metrics_direct_rate_is_zero_without_direct_questions():
    # 子集为空的分母是另一条除零通路：只跑语义题时直问率必须是 0 而不是崩掉，
    # 更不能把语义题的 exact_top1 混进来充数（那会虚高 AC-2）
    metrics = compute_metrics([_row(category="语义改写", exact_top1=True)])
    assert metrics["direct_n"] == 0
    assert metrics["direct_rate"] == 0.0


def test_render_report_states_the_ai_draft_caveat():
    text = render_report(compute_metrics([_row()]),
                         {"md5": "abc", "n": 1, "endpoint": "x"})
    assert "AI" in text and ("律师" in text or "不得作为" in text)
    assert "abc" in text, "报告必须记下评估集 MD5，否则无法证明评测的是哪一版"


def test_render_report_includes_abstain_rate():
    # 降级率是③a 最该盯的健康指标（设计文档 9.2）
    text = render_report(compute_metrics([_row(status="abstain")]),
                         {"md5": "abc", "n": 1, "endpoint": "x"})
    assert "降级率" in text


def test_render_report_names_the_rate_four_gate_pass_rate():
    # 用户 2026-09-28 裁决：这个量只能叫「四关通过率」，且"它不是 12.2 的引用一致
    # 比例"必须写进产物——否则下一次重跑就把这句话洗掉，被拍板不许的名字就复活了
    text = render_report(compute_metrics([_row()]), {"md5": "abc", "n": 1, "endpoint": "x"})
    assert "| 四关通过率 | 100.0% |" in text
    assert "| 引用准确率 |" not in text, "旧名字不得复活：它与 12.2 的口径不是一回事"
    assert "输出引用与标准条款一致的比例" in text
    assert "gold 对齐指标" in text, "要写明 12.2 的口径本期未测、属后续迭代"


def test_render_report_shows_sample_warning_when_present():
    # 小样本跑会把正典文件名的报告写成 n 很小的读数，警告行是防误引的唯一线索；
    # 存在时必须原样进产物（此前只有生成侧注入，渲染侧两个方向都没有断言）
    warning = "⚠️ **本次为小样本调试跑（n=2，`--limit`），不是全量评测结论。**"
    text = render_report(compute_metrics([_row()]),
                         {"md5": "abc", "n": 2, "endpoint": "x",
                          "sample_warning": warning})
    assert warning in text


def test_render_report_omits_sample_warning_without_the_key():
    # 反向同样要钉：全量跑的报告里冒出小样本警告会让人以为读数不完整；
    # 无该键时警告文本与它的 ⚠️ 标记都不得出现
    text = render_report(compute_metrics([_row()]),
                         {"md5": "abc", "n": 100, "endpoint": "x"})
    assert "小样本" not in text
    assert "⚠️" not in text


def _fake_module(monkeypatch, name, **attrs):
    """把一个假模块塞进 sys.modules：eval_qa 的依赖全是延迟导入，替身要先就位。"""
    import types

    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def _fake_factory(monkeypatch, services):
    """把 eval_qa 延迟导入的 app.core.factory 换成替身，返回它收到的参数流水。"""
    import types

    calls: list[dict] = []
    module = types.ModuleType("app.core.factory")

    def build_services(**kwargs):
        calls.append(kwargs)
        return services

    module.build_services = build_services
    monkeypatch.setitem(sys.modules, "app.core.factory", module)
    return calls


def _fake_services():
    """一套假重资源：对象身份足以断言「工厂的返回值被真接上了」。"""
    import types
    return types.SimpleNamespace(conn=object(), client=object(), encoder=object(),
                                 reranker=object(), answerer=object())


def _stub_hand_wired_assembly(monkeypatch):
    """手写装配那四个模块的替身：只为「有人把它改回来」的形态兜底 —— 没有它们，
    被改回的 connect()/load_model() 会去连真容器、加载 2.3GB 模型。正常形态
    （走共享工厂）下一次都不会被用到，所以它们的替身恒返回 object() 就够了。
    """
    _fake_module(monkeypatch, "app.db.mysql", connect=lambda: object())
    _fake_module(monkeypatch, "app.db.milvus", get_client=lambda: object())
    _fake_module(monkeypatch, "app.ingest.embed", load_model=lambda: object())
    _fake_module(monkeypatch, "app.retrieval.rerank", load_reranker=lambda: object())


def test_retrieval_only_mode_writes_the_latency_artifact(monkeypatch, tmp_path):
    # --retrieval-only 分支只准写延迟产物、不碰主报告：把它改成写 REPORT_PATH
    # （一次调试跑就把主报告覆盖成延迟记录）必须当场变红。依赖全打假替身
    import types

    import eval_qa

    services = _fake_services()
    factory_calls = _fake_factory(monkeypatch, services)
    _stub_hand_wired_assembly(monkeypatch)
    _fake_module(monkeypatch, "app.retrieval.pipeline",
                 retrieve=lambda question, **kwargs: None)
    _fake_module(monkeypatch, "torch",
                 cuda=types.SimpleNamespace(get_device_name=lambda index: "FakeGPU"))
    monkeypatch.setattr(eval_qa, "load_items", lambda: [{"query": "甲"}])
    monkeypatch.setattr(eval_qa, "eval_md5", lambda: "abc")
    monkeypatch.setattr(sys, "argv", ["eval_qa.py", "--retrieval-only"])
    latency, report = tmp_path / "retrieval_latency.md", tmp_path / "eval_report.md"
    # 预置一条旧离群行：跑完必须还在——这就是"扛住重跑"的那条通路
    outlier = "| 2026-09-28 13:05 | 100 | 1.004s | 4.620s | 23 | 手工补录 |"
    latency.write_text(f"{LATENCY_HISTORY_START}\n{LATENCY_HISTORY_HEADER}\n{outlier}\n"
                       f"{LATENCY_HISTORY_END}\n", encoding="utf-8")
    monkeypatch.setattr(eval_qa, "LATENCY_PATH", latency)
    monkeypatch.setattr(eval_qa, "REPORT_PATH", report)

    assert eval_qa.main() == 0
    text = latency.read_text(encoding="utf-8")
    assert "FakeGPU" in text, "运行条件要进产物，否则读数无法归因"
    assert "qa_eval_v0.jsonl" in text, "评估集文件名是 main 传进去的额外交接，缺了要红"
    assert outlier in text, "旧离群行必须被保住"
    assert not report.exists(), "本模式不得碰主报告"
    # 两个模式都必须走共享工厂：手写 connect()/load_model() 时流水为空，而
    # with_extras=True 会额外调 DeepSeek（花钱且不属于本报告的任何指标）
    assert factory_calls == [{"with_extras": False}], factory_calls


def test_main_takes_the_answerer_from_the_shared_factory(monkeypatch, tmp_path):
    """主路径的 Answerer 必须**取自共享工厂**，不是为评测另装一套。

    判据放在「_evaluate_one 收到的 answerer 是不是工厂返回的那个对象」上：
    只断言「工厂被调用过」时，把返回值丢掉、再用 connect()/load_model() 另装
    一套也照样绿 —— 而那一套不读 FL_*、不跑 validate()，会出现「换了模型目录、
    评测却用另一套模型出数」且报告里看不出来的漂移。
    """
    import eval_qa

    services = _fake_services()
    factory_calls = _fake_factory(monkeypatch, services)
    _stub_hand_wired_assembly(monkeypatch)
    monkeypatch.setattr(eval_qa, "load_items",
                        lambda: [{"id": "q1", "query": "甲", "side": "internal"}])
    monkeypatch.setattr(eval_qa, "eval_md5", lambda: "abc")
    monkeypatch.setattr(sys, "argv", ["eval_qa.py"])
    seen: list[object] = []

    def fake_evaluate_one(answerer, item, log):
        seen.append(answerer)
        return {"id": item["id"], "category": "条款号直问", "gold": [584],
                "recalled_20": [584], "recalled_50": [584], "top3": [584],
                "status": "ok", "cite_ok": True, "exact_top1": True,
                "failures": [], "latency": 0.1}

    monkeypatch.setattr(eval_qa, "_evaluate_one", fake_evaluate_one)
    monkeypatch.setattr(eval_qa, "REPORT_PATH", tmp_path / "eval_report.md")
    assert eval_qa.main() == 0
    assert seen == [services.answerer], "评测的 Answerer 必须来自共享工厂"
    assert factory_calls == [{"with_extras": False}], factory_calls
    assert (tmp_path / "eval_report.md").exists(), "主路径的报告仍要落盘"
