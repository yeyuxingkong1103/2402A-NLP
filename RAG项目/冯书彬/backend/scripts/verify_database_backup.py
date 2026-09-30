import argparse
import os
import subprocess
from pathlib import Path

from sqlalchemy.engine import URL, make_url


def build_restore_command(database_url: str, backup_path: Path) -> tuple[list[str], dict[str, str]]:
    url = _require_mysql_url(database_url)
    command = [
        "mysql",
        "--host",
        url.host or "127.0.0.1",
        "--port",
        str(url.port or 3306),
        "--user",
        url.username or "",
        url.database or "",
    ]
    environment = os.environ.copy()
    if url.password:
        environment["MYSQL_PWD"] = url.password
    return command, environment


def restore_backup(database_url: str, backup_path: Path, dry_run: bool = False) -> int:
    command, environment = build_restore_command(database_url, backup_path)
    if dry_run:
        print(f"dry-run: would restore {backup_path} into the configured independent database")
        return 0
    if not backup_path.is_file():
        raise FileNotFoundError(f"backup file does not exist: {backup_path}")
    with backup_path.open("rb") as backup_file:
        subprocess.run(command, stdin=backup_file, check=True, env=environment)
    print(f"MySQL backup restored: {backup_path}")
    return 0


def _require_mysql_url(database_url: str) -> URL:
    url = make_url(database_url)
    if url.get_backend_name() != "mysql":
        raise ValueError("DATABASE_URL must use a MySQL backend")
    if not url.database:
        raise ValueError("DATABASE_URL must include an independent verification database")
    if url.database.lower() in {"mysql", "legal_rag", "myrag", "production"}:
        raise ValueError("verification database must be explicitly independent")
    return url


def main() -> int:
    parser = argparse.ArgumentParser(description="恢复并验证 MySQL 备份到独立临时库")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return restore_backup(args.database_url, args.backup, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
