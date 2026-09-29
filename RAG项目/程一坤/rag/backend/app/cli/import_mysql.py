import argparse
import json
import os
from pathlib import Path

from app.db import sql_models  # noqa: F401  # 导入模型以注册 Base.metadata
from app.db.base import Base
from app.db.batch_import import import_packages
from app.db.engine import create_database_engine, create_session_factory
from app.db.import_service import import_package
from app.pipeline.package_validator import PackageValidationError, validate_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导入已校验的法律数据包到 MySQL。")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--package", help="单个数据包目录路径")
    group.add_argument("--packages-root", help="数据包根目录路径（批量导入）")
    parser.add_argument("--database-url", help="数据库连接串；未提供时读取 DATABASE_URL")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="强制重建：删除已存在文档的旧版本与旧 chunk，再导入新解析结果（事务内执行）",
    )
    args = parser.parse_args(argv)

    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        parser.exit(2, "database url is required\n")

    engine = create_database_engine(database_url)
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)

    if args.package:
        return _import_single_package(args.package, session_factory, parser, args.replace_existing)
    else:
        return _import_batch_packages(args.packages_root, session_factory, parser, args.replace_existing)


def _import_single_package(package_path: str, session_factory, parser, replace_existing: bool) -> int:
    package_directory = Path(package_path)
    if not package_directory.is_dir():
        parser.exit(2, "package directory does not exist\n")

    try:
        validated_package = validate_package(package_directory)
        with session_factory() as session:
            result = import_package(session, validated_package, replace_existing=replace_existing)
    except PackageValidationError as error:
        _print_import_result({"status": "failed", "stage": "validate", "error": str(error)})
        return 1
    except Exception as error:
        _print_import_result({"status": "failed", "stage": "import", "error": error.__class__.__name__})
        return 1

    _print_import_result(
        {
            "status": result.status,
            "package_id": result.package_id,
            "version_status": result.version_status,
            "document_key": result.document_key,
            "version_key": result.version_key,
        }
    )
    return 0 if result.status in {"imported", "already_imported"} else 1


def _import_batch_packages(packages_root: str, session_factory, parser, replace_existing: bool) -> int:
    packages_directory = Path(packages_root)
    if not packages_directory.is_dir():
        parser.exit(2, "packages directory does not exist\n")

    try:
        with session_factory() as session:
            result = import_packages(session, packages_directory, replace_existing=replace_existing)
    except FileNotFoundError as error:
        parser.exit(2, f"{error}\n")
    except Exception as error:
        _print_import_result({"status": "failed", "stage": "batch_import", "error": error.__class__.__name__})
        return 1

    _print_import_result(
        {
            "status": "imported" if result.failed == 0 else "partial_failed",
            "total_packages": result.total_packages,
            "imported": result.imported,
            "already_imported": result.already_imported,
            "failed": result.failed,
        }
    )
    return 0 if result.failed == 0 else 1


def _print_import_result(payload: dict[str, str | None]) -> None:
    """只输出脱敏结构化结果，不回显数据库连接串或正文。"""
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
