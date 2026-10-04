# 配置装载只做三件事（设计 §三）：读环境变量、给默认值、失效时报错。
# 三条各配一组断言，另加一条「本机真路径可用」——它是 tools/check_env.py 的
# 离线影子：那条自检要连库/看 GPU，而这条只问「默认路径下装配所需文件在不在」。
import pytest

from app.core import config, errors
from app.db.milvus import MILVUS_URI
from app.db.mysql import DEFAULT_CONFIG
from app.ingest.embed import DEFAULT_MODEL_PATH
from app.retrieval.rerank import RERANK_MODEL_PATH


def test_defaults_come_from_the_single_sources():
    """默认值必须**取自**各处的既有常量，不是抄一份字面量。

    抄了就会改一处漏一处：milvus / mysql 那两处改了而这里没跟上时，CLI（直连
    connect）与 HTTP（走 config）会连到不同的地方，而两边各自的测试都是绿的。
    """
    cfg = config.load(env={})
    assert cfg.milvus_uri == MILVUS_URI
    assert cfg.embed_model_path == DEFAULT_MODEL_PATH
    assert cfg.reranker_model_path == RERANK_MODEL_PATH
    for key, value in DEFAULT_CONFIG.items():
        assert cfg.mysql[key] == value


def test_env_overrides_win_and_untouched_fields_keep_defaults():
    cfg = config.load(env={config.ENV_MILVUS_URI: "http://db.example:19530",
                           config.ENV_EMBED_MODEL_PATH: r"X:\m1",
                           config.ENV_RERANK_MODEL_PATH: r"X:\m2",
                           "FL_MYSQL_HOST": "db.example", "FL_MYSQL_PORT": "3307"})
    assert cfg.milvus_uri == "http://db.example:19530"
    assert cfg.embed_model_path == r"X:\m1"
    assert cfg.reranker_model_path == r"X:\m2"
    assert cfg.mysql["host"] == "db.example"
    # 端口必须是 int：pymysql 收到字符串时会在驱动内部某处比较大小并抛 TypeError，
    # 报错点离「谁把这个值配错了」很远，故在本层就转
    assert cfg.mysql["port"] == 3307 and isinstance(cfg.mysql["port"], int)
    assert cfg.mysql["user"] == DEFAULT_CONFIG["user"], "未覆盖的字段仍取默认值"


def test_blank_env_is_treated_as_unset():
    # .env 里 `FL_MILVUS_URI=` 这种空行不该把服务指向一个空地址——那看起来像配好了
    cfg = config.load(env={config.ENV_MILVUS_URI: "   ", "FL_MYSQL_PASSWORD": ""})
    assert cfg.milvus_uri == MILVUS_URI
    assert cfg.mysql["password"] == DEFAULT_CONFIG["password"]


def test_a_non_numeric_port_is_rejected_at_load_time():
    # 报错点必须在本层：走到连接期才炸的话，现场只剩一句不带字段名的 ValueError
    with pytest.raises(config.ConfigError, match="FL_MYSQL_PORT"):
        config.load(env={"FL_MYSQL_PORT": "3306a"})


def _dir_with(tmp_path, name: str, *filenames: str) -> str:
    """造一个含指定文件的目录，返回路径字符串。"""
    path = tmp_path / name
    path.mkdir()
    for filename in filenames:
        (path / filename).write_text("placeholder", encoding="utf-8")
    return str(path)


def _config_for(tmp_path, *, embed=("sparse_linear.pt",),
                rerank=("model.safetensors",)) -> config.Config:
    return config.Config(milvus_uri=MILVUS_URI, mysql=dict(DEFAULT_CONFIG),
                         embed_model_path=_dir_with(tmp_path, "embed", *embed),
                         reranker_model_path=_dir_with(tmp_path, "rerank", *rerank))


def test_validate_passes_when_the_required_files_are_present(tmp_path):
    _config_for(tmp_path).validate()  # 不抛即通过


def test_validate_reports_every_missing_model_file_at_once(tmp_path):
    """缺件要当场抛，且**一次列全**：换机器 / 挂盘时缺件常是成片的，
    一次看全比重跑几次各看一条便宜。两个文件名都进来，说明它没在第一条上短路。"""
    with pytest.raises(config.ConfigError) as exc:
        _config_for(tmp_path, embed=(), rerank=()).validate()
    message = str(exc.value)
    assert "sparse_linear.pt" in message and "model.safetensors" in message


def test_default_paths_are_valid_on_this_machine():
    """默认指向的两个模型目录在本机必须真能用（真依赖优先的项目惯例）。

    它挡的是「默认值指向一个不存在的目录」这类改坏：那种坏法在单测里全绿，
    直到有人真跑 CLI 才炸在加载期。
    """
    config.load(env={}).validate()


# ---- 应用层防滥用（任务 6）：限流阈值与加盐值 ----

def test_rate_limit_defaults_are_the_ones_the_design_and_the_report_named():
    """四个默认值当字面量锚：并发 20 是设计 §六 的数字，其余三个是任务 6 的裁决值。

    写成字面量而不是引用 config.DEFAULT_*：后者在有人改默认值时恒绿，而「不设
    环境变量时到底限多少」正是运维要能一眼查到的事（报告里给了算账）。
    """
    settings = config.load_rate_limit(env={})
    assert settings.concurrency == 20, "设计 §六：公众侧同时处理数 ≤ 20（AC-14）"
    assert settings.limit == 30
    assert settings.window_s == 60.0
    assert settings.max_keys == 10_000


def test_rate_limit_env_overrides_win():
    settings = config.load_rate_limit(env={
        config.ENV_RATE_LIMIT_MAX: "5",
        config.ENV_RATE_LIMIT_WINDOW_S: "2.5",
        config.ENV_RATE_LIMIT_MAX_KEYS: "7",
        config.ENV_PUBLIC_CONCURRENCY: "3"})
    assert (settings.limit, settings.window_s, settings.max_keys,
            settings.concurrency) == (5, 2.5, 7, 3)
    assert isinstance(settings.limit, int) and isinstance(settings.window_s, float)


@pytest.mark.parametrize("name", ["FL_RATE_LIMIT_MAX", "FL_RATE_LIMIT_WINDOW_S",
                                  "FL_RATE_LIMIT_MAX_KEYS", "FL_PUBLIC_CONCURRENCY"])
def test_a_non_numeric_rate_limit_value_is_rejected_at_load_time(name):
    """填了非数字当场抛，**不退回默认值**：退回的形态是「运维以为限了 5 次，
    实际是 30 次」，没有任何红灯 —— 阈值与模型路径不同，必须响亮地失败。"""
    with pytest.raises(config.ConfigError, match=name):
        config.load_rate_limit(env={name: "not-a-number"})


def test_blank_rate_limit_values_are_treated_as_unset():
    # 与 _text 同口径：.env 里 `FL_RATE_LIMIT_MAX=` 这样的空行不该把阈值清零
    assert config.load_rate_limit(env={config.ENV_RATE_LIMIT_MAX: "  "}).limit == 30


def test_a_missing_or_blank_salt_is_rejected():
    """**没有默认盐**（设计 §六 的「不存 IP 原文」只有配了盐才成立）。

    固定默认盐等于把 IP 哈希重新变成可反推的（2^32 的枚举量对一台笔记本是几分钟）。
    """
    for env in ({}, {config.ENV_RATE_LIMIT_SALT: ""},
                {config.ENV_RATE_LIMIT_SALT: "   "}):
        with pytest.raises(config.MissingRateSaltError,
                           match=config.ENV_RATE_LIMIT_SALT):
            config.load_rate_salt(env=env)


def test_a_short_salt_is_rejected_and_the_message_does_not_echo_the_value():
    """盐过短也拒（枚举反推的成本随长度指数下降），且报错不回显取值。"""
    with pytest.raises(config.MissingRateSaltError, match="过短") as exc:
        config.load_rate_salt(env={config.ENV_RATE_LIMIT_SALT: "short"})
    assert "short" not in str(exc.value)


def test_a_long_enough_salt_is_returned_verbatim():
    salt = "s" * config.MIN_RATE_SALT_LEN
    assert config.load_rate_salt(env={config.ENV_RATE_LIMIT_SALT: salt}) == salt


def test_the_missing_salt_error_belongs_to_the_config_error_family():
    """它是 ConfigError 的兄弟而不是 ApiError：配置故障永远映射不到 429/404。

    这条钉的是**类型谱系**（用例直接判 issubclass）：若哪天有人把它改成
    ApiError 的子类，缺盐就会以某个业务状态码的面貌出现 —— 那正是「配置故障
    不许伪装成业务拒绝」要防的形态，且改错的当时不会有别的红灯。
    """
    assert issubclass(config.MissingRateSaltError, config.ConfigError)
    assert not issubclass(config.MissingRateSaltError, errors.ApiError)
