import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import write_package


def run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)
        for key in ("DATABASE_URL", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE"):
            if key not in env:
                process_env.pop(key, None)
    return subprocess.run(
        [sys.executable, "-m", "app.cli.index_legal_documents", *args],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
        env=process_env,
    )


def test_index_legal_documents_cli_rejects_missing_database_url() -> None:
    result = run_cli("--limit", "1", env={})

    assert result.returncode == 2
    assert "database url is required" in result.stderr




def mark_versions_approved(database_path) -> None:
    """模拟审核通过：导入后把版本置为 approved（阶段6 起索引只处理已发布版本）。"""
    import sqlite3

    conn = sqlite3.connect(str(database_path))
    conn.execute("UPDATE document_versions SET version_status='approved'")
    conn.commit()
    conn.close()


def test_index_legal_documents_cli_accepts_embedding_batch_size(tmp_path: Path) -> None:
    package_directory = tmp_path / "pkg-cli-batch"
    package_directory.mkdir()
    database_path = tmp_path / "cli-batch.db"
    write_package(package_directory)

    import_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli.import_mysql",
            "--package",
            str(package_directory),
            "--database-url",
            f"sqlite+pysqlite:///{database_path.as_posix()}",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert import_result.returncode == 0
    mark_versions_approved(database_path)

    result = run_cli(
        "--limit",
        "1",
        "--embedding-batch-size",
        "1",
        "--database-url",
        f"sqlite+pysqlite:///{database_path.as_posix()}",
        "--fake-services",
        env={},
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "indexed_chunks": 1,
        "indexed_versions": 1,
        "status": "indexed",
    }


def test_index_legal_documents_cli_indexes_sqlite_rows_with_fake_services(tmp_path: Path) -> None:
    package_directory = tmp_path / "pkg-cli"
    package_directory.mkdir()
    database_path = tmp_path / "cli.db"
    write_package(package_directory)
    import_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli.import_mysql",
            "--package",
            str(package_directory),
            "--database-url",
            f"sqlite+pysqlite:///{database_path.as_posix()}",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert import_result.returncode == 0
    mark_versions_approved(database_path)

    result = run_cli(
        "--limit",
        "1",
        "--database-url",
        f"sqlite+pysqlite:///{database_path.as_posix()}",
        "--fake-services",
        env={},
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload == {"indexed_chunks": 1, "indexed_versions": 1, "status": "indexed"}
    assert "sqlite" not in result.stdout
