# 环境自检的测试只验"检查项是否齐全、结论是否如实表达"，
# 不验具体环境好不好——那由 tools/check_env.py 实跑输出证明。
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import check_env
from check_env import CHECKS, check_all


def test_check_names_cover_the_contracts_this_stage_signed():
    # 每一项都对应一个既定承诺：容器要起、GPU 要能跑精排、密钥要能被读到。
    # 少一项就意味着某个承诺没人验。reranker 是第六项，最容易在改名时被漏掉——
    # 少了它，精排模型目录缺失只会让检索安静退化，不会报错。
    names = [name for name, _, _ in check_all()]
    for required in ("cuda", "milvus", "mysql", "ollama", "deepseek_key", "reranker"):
        assert required in names, f"自检缺少 {required}"


def test_check_all_returns_triples():
    for item in check_all():
        assert len(item) == 3
        name, ok, detail = item
        assert isinstance(name, str) and name
        assert isinstance(ok, bool)
        assert isinstance(detail, str)


def test_checks_registry_is_not_empty():
    assert len(CHECKS) >= 5


def _model_dirs(tmp_path):
    """造两个含必需文件的模型目录，返回 (编码器目录, 精排目录)。

    必需文件名刻意不写在这里的断言外：它们由 core.config 的校验唯一持有，
    本测试只用「缺了会不会报」来验委托关系，不再自己抄一份清单。
    """
    embed, rerank = tmp_path / "embed", tmp_path / "rerank"
    for directory, name in ((embed, "sparse_linear.pt"), (rerank, "model.safetensors")):
        directory.mkdir()
        (directory / name).write_text("placeholder", encoding="utf-8")
    return embed, rerank


def test_model_check_delegates_to_the_config_and_honours_fl_overrides(monkeypatch, tmp_path):
    """自检的模型目录必须与真装配同源：把 FL_* 覆盖指到临时目录，读数要跟着变。

    自己抄一份路径的实现在这条上恒绿 —— 它看的是仓库默认目录（本机存在），
    于是「换了模型目录」对自检完全不可见，而真装配已经换了，两侧各看各的。
    """
    embed, rerank = _model_dirs(tmp_path)
    monkeypatch.setenv("FL_EMBED_MODEL_PATH", str(embed))
    monkeypatch.setenv("FL_RERANK_MODEL_PATH", str(rerank))
    ok, detail = check_env._check_reranker()
    assert ok is True, detail
    assert str(embed) in detail and str(rerank) in detail


def test_model_check_names_each_missing_file(monkeypatch, tmp_path):
    """缺件必须报未通过、且点名缺的是谁。两个目录各测一次。

    编码器那半是**委托校验顺带扩出来的**：只查精排的版本在它上面恒绿（默认目录
    本机存在）—— 而稀疏头缺件不报错、只安静退化，正是自检该拦的那类。反向那次
    拦「校验退化成只看一个目录」。
    """
    embed, rerank = _model_dirs(tmp_path)
    monkeypatch.setenv("FL_EMBED_MODEL_PATH", str(embed))
    monkeypatch.setenv("FL_RERANK_MODEL_PATH", str(rerank))
    (embed / "sparse_linear.pt").unlink()
    ok, detail = check_env._check_reranker()
    assert ok is False and "sparse_linear.pt" in detail, detail
    (rerank / "model.safetensors").unlink()
    ok, detail = check_env._check_reranker()
    assert ok is False and "model.safetensors" in detail, detail


def test_check_all_reports_failure_when_a_check_raises(monkeypatch):
    # 自检的全部价值就是"环境不对就说不对"。若 check_all 把异常吞成通过，
    # 容器全掉也会打印 6/6 通过，比没有自检更误导（③a 后期就是这么踩坑的）。
    # 所以这条专钉异常分支：它一旦被改成谎报成功，本测试必须变红。
    def boom() -> tuple[bool, str]:
        raise RuntimeError("模拟容器掉线")

    def fine() -> tuple[bool, str]:
        return True, "ok"

    # 后面还挂一个正常项：一个检查炸掉不能带走整轮，它之后的项仍要跑到。
    monkeypatch.setattr(check_env, "CHECKS", [("boom", boom), ("fine", fine)])
    results = check_all()
    assert [name for name, _, _ in results] == ["boom", "fine"]
    name, ok, detail = results[0]
    assert name == "boom"
    assert ok is False, "检查函数抛异常时必须报未通过，不能谎报成功"
    assert detail, "异常原因必须落进说明里，空说明等于没验"
    assert "模拟容器掉线" in detail, f"说明里要带异常原文，实际：{detail!r}"


def test_main_exit_code_reflects_failed_checks(monkeypatch):
    # main() 的返回码是"能不能开工"的唯一信号，CI 与验收只看它：
    # 恒定 0 会把坏环境放行，恒定 1 会让绿灯永远等不到，两个方向都要钉住。
    # 这里用"抛异常"的检查项而非返回 (False, ...)，是为了顺带把
    # "异常 → 未通过 → 退出码 1" 这条端到端路径也钉上。
    def boom() -> tuple[bool, str]:
        raise RuntimeError("模拟容器掉线")

    monkeypatch.setattr(check_env, "CHECKS", [("boom", boom)])
    assert check_env.main() == 1, "有未通过项（含异常）时退出码必须是 1"
    # 反向也要验：只测失败方向的话，把 main 写成恒返回 1 也照样全绿
    monkeypatch.setattr(check_env, "CHECKS", [("always_pass", lambda: (True, "ok"))])
    assert check_env.main() == 0, "全部通过时退出码必须是 0"
