import json
import subprocess
import sys
from pathlib import Path

from conftest import write_package


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "app.cli.import_mysql", *args],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )


def test_import_mysql_cli_rejects_missing_package(tmp_path: Path) -> None:
    result = run_cli(
        "--package",
        str(tmp_path / "missing"),
        "--database-url",
        "sqlite+pysqlite:///:memory:",
    )

    assert result.returncode == 2
    assert "package directory does not exist" in result.stderr


def test_import_mysql_cli_rejects_invalid_package(tmp_path: Path) -> None:
    bad_package = tmp_path / "bad-package"
    bad_package.mkdir()

    result = run_cli(
        "--package",
        str(bad_package),
        "--database-url",
        "sqlite+pysqlite:///:memory:",
    )

    assert result.returncode == 1
    assert "failed" in result.stdout
    assert "第一条 测试正文" not in result.stdout + result.stderr


def test_import_mysql_cli_imports_valid_package(tmp_path: Path) -> None:
    package_directory = tmp_path / "pkg-cli"
    package_directory.mkdir()
    database_path = tmp_path / "cli.db"
    write_package(package_directory)

    result = run_cli(
        "--package",
        str(package_directory),
        "--database-url",
        f"sqlite+pysqlite:///{database_path.as_posix()}",
    )

    assert result.returncode == 0
    assert "imported" in result.stdout
    assert "pkg-cli" in result.stdout
    assert "sqlite" not in result.stdout
