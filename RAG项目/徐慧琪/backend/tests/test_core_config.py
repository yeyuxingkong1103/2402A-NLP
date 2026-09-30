# 配置装载只做三件事（设计 §三）：读环境变量、给默认值、失效时报错。
# 三条各配一组断言，另加一条「本机真路径可用」——它是 tools/check_env.py 的
# 离线影子：那条自检要连库/看 GPU，而这条只问「默认路径下装配所需文件在不在」。
import pytest

from app.core import config
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
