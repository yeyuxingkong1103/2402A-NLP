"""批次 33/34：按 ENVIRONMENT 选择 .env 文件的加载逻辑测试。

覆盖 `app.core.config.load_environment_file` 的文件选择与优先级规则：
- ENVIRONMENT=production 且 .env.production 缺失 → 明确报错（不许静默回退默认值）
- ENVIRONMENT 取值未知（拼写错误）→ 明确报错
- development/test：先读 `.env.<env>`，再读本机私有 `.env` **覆盖**其刚写入的键
  （批次 34 裁决：三份交付文件全占位符，真实凭据在私有 .env 里，必须能盖回来）
- 进程环境变量优先级最高：任何文件都不覆盖进程已有变量
- 显式传入 file_path 时行为不变（既有工具脚本与测试用法，单文件 setdefault）
- 已存在的进程环境变量不被覆盖（既有语义保留）

注意：本模块 import `app.core.config` 时会触发模块级 `settings` 实例化——
本机开发环境 ENVIRONMENT 未设置 → development → 读真实 .env，
与改动前行为一致，不会触发 production 必填校验。
"""

from pathlib import Path

import pytest

from app.core import config as config_module


@pytest.fixture()
def env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把配置目录指到 tmp_path，让用例在受控目录里摆放 .env 文件。"""
    root = tmp_path / "envroot"
    root.mkdir()
    monkeypatch.setattr(config_module, "ENVIRONMENT_FILE_PATH", root / ".env")
    return root


def _unset(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    """删掉进程环境里可能干扰用例的变量（monkeypatch 自动还原）。"""
    monkeypatch.delenv(key, raising=False)


def test_production_missing_file_raises(env_root: Path, monkeypatch) -> None:
    """ENVIRONMENT=production 但缺 .env.production → 必须明确报错。"""
    _unset(monkeypatch, "FOO")
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(RuntimeError) as excinfo:
        config_module.load_environment_file()
    assert ".env.production" in str(excinfo.value)


def test_unknown_environment_raises(env_root: Path, monkeypatch) -> None:
    """ENVIRONMENT 拼写错误（如 staging）→ 明确报错，不静默落回 development。"""
    monkeypatch.setenv("ENVIRONMENT", "staging")
    with pytest.raises(RuntimeError) as excinfo:
        config_module.load_environment_file()
    assert "staging" in str(excinfo.value)


def test_development_falls_back_to_dotenv(env_root: Path, monkeypatch) -> None:
    """development：无 .env.development → 只读旧单文件 .env（等价回退）。"""
    _unset(monkeypatch, "FOO")
    monkeypatch.setenv("ENVIRONMENT", "development")
    (env_root / ".env").write_text("FOO=from-dotenv\n", encoding="utf-8")
    loaded = config_module.load_environment_file()
    assert loaded == 1
    import os

    assert os.environ["FOO"] == "from-dotenv"


def test_only_env_envfile_takes_its_value(env_root: Path, monkeypatch) -> None:
    """只有 .env.<env> 没有 .env → 取 .env.<env> 的值（占位符场景）。"""
    _unset(monkeypatch, "FOO")
    monkeypatch.setenv("ENVIRONMENT", "development")
    (env_root / ".env.development").write_text("FOO=from-dev-file\n", encoding="utf-8")
    loaded = config_module.load_environment_file()
    assert loaded == 1
    import os

    assert os.environ["FOO"] == "from-dev-file"


def test_private_env_overrides_env_envfile(env_root: Path, monkeypatch) -> None:
    """批次 34 核心语义：.env 后读并覆盖 .env.<env> 刚写入的键。"""
    _unset(monkeypatch, "FOO")
    monkeypatch.setenv("ENVIRONMENT", "development")
    (env_root / ".env.development").write_text("FOO=from-dev-file\n", encoding="utf-8")
    (env_root / ".env").write_text("FOO=from-dotenv\n", encoding="utf-8")
    loaded = config_module.load_environment_file()
    assert loaded == 2  # 主文件写 1 次 + 私有文件覆盖 1 次
    import os

    assert os.environ["FOO"] == "from-dotenv"


def test_placeholder_overridden_by_private_env(env_root: Path, monkeypatch) -> None:
    """用户指定用例：.env.<env> 占位符 + .env 真值 → 最终取到真值。"""
    _unset(monkeypatch, "MYSQL_USER")
    monkeypatch.setenv("ENVIRONMENT", "development")
    (env_root / ".env.development").write_text(
        "MYSQL_USER=<your-mysql-user>\n", encoding="utf-8"
    )
    (env_root / ".env").write_text("MYSQL_USER=real_user\n", encoding="utf-8")
    loaded = config_module.load_environment_file()
    import os

    assert os.environ["MYSQL_USER"] == "real_user"
    assert loaded == 2


def test_process_env_beats_both_files(env_root: Path, monkeypatch) -> None:
    """进程环境变量优先级最高：两份文件都写同名键也盖不掉。"""
    monkeypatch.setenv("FOO", "from-process")
    monkeypatch.setenv("ENVIRONMENT", "development")
    (env_root / ".env.development").write_text("FOO=from-dev-file\n", encoding="utf-8")
    (env_root / ".env").write_text("FOO=from-dotenv\n", encoding="utf-8")
    loaded = config_module.load_environment_file()
    assert loaded == 0
    import os

    assert os.environ["FOO"] == "from-process"


def test_environment_key_not_overridden_by_private_file(
    env_root: Path, monkeypatch
) -> None:
    """ENVIRONMENT 是进程变量时不被 .env 覆盖（防"生产误降级为开发"），
    但主文件写入的普通键仍可被 .env 覆盖。"""
    monkeypatch.setenv("ENVIRONMENT", "production")
    (env_root / ".env.production").write_text(
        "PRIMARY_ONLY=x\n", encoding="utf-8"
    )
    (env_root / ".env").write_text(
        "ENVIRONMENT=development\nPRIMARY_ONLY=y\n", encoding="utf-8"
    )
    loaded = config_module.load_environment_file()
    import os

    assert os.environ["ENVIRONMENT"] == "production"  # 进程变量未被覆盖
    assert os.environ["PRIMARY_ONLY"] == "y"  # 主文件的键被私有文件覆盖
    assert loaded == 2


def test_both_missing_returns_zero(env_root: Path, monkeypatch) -> None:
    """development：两份文件都缺 → 返回 0（默认值 + 进程环境变量可用）。"""
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert config_module.load_environment_file() == 0


def test_test_env_reads_env_test(env_root: Path, monkeypatch) -> None:
    """ENVIRONMENT=test → 读 .env.test。"""
    _unset(monkeypatch, "FOO")
    monkeypatch.setenv("ENVIRONMENT", "test")
    (env_root / ".env.test").write_text("FOO=from-test-file\n", encoding="utf-8")
    loaded = config_module.load_environment_file()
    assert loaded == 1
    import os

    assert os.environ["FOO"] == "from-test-file"


def test_explicit_path_still_works(env_root: Path, monkeypatch) -> None:
    """显式传入 file_path → 直接读该文件，与环境名无关。"""
    _unset(monkeypatch, "FOO")
    monkeypatch.setenv("ENVIRONMENT", "production")  # 即便 production 也不走选择逻辑
    explicit = env_root / "custom.env"
    explicit.write_text("FOO=explicit\n", encoding="utf-8")
    loaded = config_module.load_environment_file(file_path=explicit)
    assert loaded == 1
    import os

    assert os.environ["FOO"] == "explicit"


def test_existing_env_var_not_overridden(env_root: Path, monkeypatch) -> None:
    """文件里的键不覆盖已存在的环境变量（既有语义保留）。"""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("FOO", "already-set")
    (env_root / ".env.development").write_text("FOO=from-file\n", encoding="utf-8")
    loaded = config_module.load_environment_file()
    assert loaded == 0
    import os

    assert os.environ["FOO"] == "already-set"
