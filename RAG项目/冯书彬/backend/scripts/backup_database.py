import argparse
import os
import subprocess
from pathlib import Path

from sqlalchemy.engine import URL, make_url


DUMP_OPTIONS = (
    "--single-transaction",
    "--routines",
    "--events",
    "--hex-blob",
    "--skip-extended-insert",
)


def build_dump_command(database_url: str, output_path: Path) -> tuple[list[str], dict[str, str]]:
    url = _require_mysql_url(database_url)
    command = [
        "mysqldump",
        *DUMP_OPTIONS,
        "--host",
        url.host or "127.0.0.1",
        "--port",
        str(url.port or 3306),
        "--user",
        url.username or "",
        "--result-file",
        str(output_path),
        url.database or "",
    ]
    environment = os.environ.copy()
    if url.password:
        environment["MYSQL_PWD"] = url.password
    return command, environment


def create_backup(database_url: str, output_path: Path, dry_run: bool = False) -> int:
    command, environment = build_dump_command(database_url, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"dry-run: would write MySQL backup to {output_path}")
        return 0
    subprocess.run(command, check=True, env=environment)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("mysqldump completed without a non-empty backup file")
    print(f"MySQL backup created: {output_path} ({output_path.stat().st_size} bytes)")
    return 0


def _require_mysql_url(database_url: str) -> URL:
    url = make_url(database_url)
    if url.get_backend_name() != "mysql":
        raise ValueError("DATABASE_URL must use a MySQL backend")
    if not url.database:
        raise ValueError("DATABASE_URL must include a database name")
    return url


def main() -> int:
    parser = argparse.ArgumentParser(description="创建一致性 MySQL 备份")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return create_backup(args.database_url, args.output, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
