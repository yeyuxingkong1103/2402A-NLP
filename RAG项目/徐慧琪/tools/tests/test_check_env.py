# 环境自检的测试只验"检查项是否齐全、结论是否如实表达"，
# 不验具体环境好不好——那由 tools/check_env.py 实跑输出证明。
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import check_env
from check_env import CHECKS, check_all


def test_check_names_cover_the_contracts_this_stage_signed():
    # 每一项都对应一个既定承诺：容器要起、GPU 要能跑精排、密钥与限流盐要能被读到。
    # 少一项就意味着某个承诺没人验。reranker 是最容易在改名时被漏掉的老项——
    # 少了它，精排模型目录缺失只会让检索安静退化，不会报错；rate_salt 是任务 6
    # 新加的一项（缺了它公开端点全 500），同样必须有人在名单上。
    names = [name for name, _, _ in check_all()]
    for required in ("cuda", "milvus", "mysql", "ollama", "deepseek_key", "reranker",
                     "rate_salt", "jwt_secret"):
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


def test_rate_salt_check_fails_loudly_when_the_env_var_is_missing(monkeypatch):
    """缺盐当场报未通过、点名缺的变量；过短也报未通过（任务 6 的部署缺口前移）。

    缺盐的后果是公开端点**全部 500**，与 deepseek_key 同一类「配置没到位」，
    故同一条纪律：只报存在性与长度，绝不回显取值。过短那半顺带证明判定真的
    走 config.load_rate_salt 的长度规则，而不是只看「变量非空」。
    """
    monkeypatch.delenv("FL_RATE_LIMIT_SALT", raising=False)
    ok, detail = check_env._check_rate_salt()
    assert ok is False, "缺盐必须报未通过"
    assert "FL_RATE_LIMIT_SALT" in detail, f"要点名缺的变量，实际：{detail!r}"
    monkeypatch.setenv("FL_RATE_LIMIT_SALT", "short")
    ok, detail = check_env._check_rate_salt()
    assert ok is False and "过短" in detail, f"过短的盐也必须报，实际：{detail!r}"


def test_rate_salt_check_passes_when_set_and_never_echoes_the_value(monkeypatch):
    """设上就 OK；且说明里不含盐的取值（回显等于把盐写进日志）。"""
    salt = "check-env-salt-0123456789abcdef"
    monkeypatch.setenv("FL_RATE_LIMIT_SALT", salt)
    ok, detail = check_env._check_rate_salt()
    assert ok is True, detail
    assert salt not in detail, "自检说明里不许出现盐的取值"


def test_jwt_secret_check_fails_loudly_when_the_env_var_is_missing(monkeypatch):
    """缺密钥当场报未通过、点名缺的变量；过短也报（终审 I-5）。

    缺密钥的后果是**服务起不来**（lifespan 启动自检），部署时要在起服务之前就
    拦住 —— 与 rate_salt 同一条纪律：只报存在性与长度，绝不回显取值。
    """
    monkeypatch.delenv("FL_JWT_SECRET", raising=False)
    ok, detail = check_env._check_jwt_secret()
    assert ok is False, "缺签名密钥必须报未通过"
    assert "FL_JWT_SECRET" in detail, f"要点名缺的变量，实际：{detail!r}"
    monkeypatch.setenv("FL_JWT_SECRET", "short")
    ok, detail = check_env._check_jwt_secret()
    assert ok is False and "过短" in detail, f"过短的密钥也必须报，实际：{detail!r}"


def test_jwt_secret_check_passes_when_set_and_never_echoes_the_value(monkeypatch):
    """设上就 OK；且说明里不含密钥取值（回显等于把密钥写进日志）。"""
    secret = "check-env-secret-0123456789abcdef-0123456789"
    monkeypatch.setenv("FL_JWT_SECRET", secret)
    ok, detail = check_env._check_jwt_secret()
    assert ok is True, detail
    assert secret not in detail, "自检说明里不许出现密钥取值"


def test_a_missing_jwt_secret_makes_the_cli_exit_nonzero(monkeypatch):
    """端到端：名单只挂 jwt_secret 一项时，缺密钥 main() 退出码非 0、设上则 0。"""
    monkeypatch.setattr(check_env, "CHECKS",
                        [("jwt_secret", check_env._check_jwt_secret)])
    monkeypatch.delenv("FL_JWT_SECRET", raising=False)
    assert check_env.main() == 1, "缺密钥时退出码必须非 0（部署自检要当场拦下）"
    monkeypatch.setenv("FL_JWT_SECRET", "check-env-secret-0123456789abcdef-0123456789")
    assert check_env.main() == 0, "密钥设上后退出码必须是 0"


def test_a_missing_rate_salt_makes_the_cli_exit_nonzero(monkeypatch):
    """端到端：真 CHECKS 里的 rate_salt 项缺盐时 main() 退出码非 0、设上则 0。

    名单只挂 rate_salt 一项（不跑整个 CHECKS）是为了让这条判据只依赖盐这一件事 ——
    容器/模型好不好由别的用例与环境的实跑证明。
    """
    monkeypatch.setattr(check_env, "CHECKS",
                        [("rate_salt", check_env._check_rate_salt)])
    monkeypatch.delenv("FL_RATE_LIMIT_SALT", raising=False)
    assert check_env.main() == 1, "缺盐时退出码必须非 0（部署自检要当场拦下）"
    monkeypatch.setenv("FL_RATE_LIMIT_SALT", "check-env-salt-0123456789abcdef")
    assert check_env.main() == 0, "盐设上后退出码必须是 0"


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
