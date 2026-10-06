from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.pipeline.package_validator import PackageValidationError, validate_package
from app.db.import_service import ImportPackageResult, import_package


@dataclass(frozen=True)
class BatchImportResult:
    """批量导入汇总，不保存或输出数据库连接信息。"""

    total_packages: int
    imported: int
    already_imported: int
    failed: int
    results: tuple[ImportPackageResult, ...]


def find_package_directories(packages_root: Path) -> list[Path]:
    """只返回包含 manifest 的标准数据包目录，忽略临时目录和普通文件。"""
    if not packages_root.is_dir():
        raise FileNotFoundError(f"packages directory does not exist: {packages_root.name}")
    return sorted(
        path
        for path in packages_root.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )


def import_package_directory(
    session: Session,
    package_directory: Path,
    replace_existing: bool = False,
) -> ImportPackageResult:
    """校验并导入单个标准数据包，校验失败时返回脱敏失败结果。"""
    try:
        validated = validate_package(package_directory)
    except PackageValidationError as error:
        return ImportPackageResult(
            status="failed",
            package_id=package_directory.name,
            error_summary=str(error),
        )
    return import_package(session, validated, replace_existing=replace_existing)


def import_packages(
    session: Session,
    packages_root: Path,
    replace_existing: bool = False,
) -> BatchImportResult:
    """按稳定目录顺序导入全部标准数据包，并隔离单包失败。"""
    package_directories = find_package_directories(packages_root)
    results = tuple(
        import_package_directory(session, package_directory, replace_existing)
        for package_directory in package_directories
    )
    return BatchImportResult(
        total_packages=len(results),
        imported=sum(result.status == "imported" for result in results),
        already_imported=sum(result.status == "already_imported" for result in results),
        failed=sum(result.status == "failed" for result in results),
        results=results,
    )
