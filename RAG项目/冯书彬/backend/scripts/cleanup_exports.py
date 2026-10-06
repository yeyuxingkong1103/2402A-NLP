import argparse
import logging

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="清理过期或已撤销的用户数据导出作业")
    parser.add_argument("--dry-run", action="store_true", help="仅输出当前实现限制说明，不删除数据")
    args = parser.parse_args()
    if args.dry_run:
        print("dry-run: 未连接数据库，未执行删除；正式清理请移除 --dry-run。")
        return 0
    from backend.app.core.config import settings
    from backend.app.core.storage import configure_auth_store
    from backend.app.services.export_service import cleanup_expired_export_jobs

    configure_auth_store(settings)
    deleted = cleanup_expired_export_jobs()
    print(f"导出作业清理完成：exports={deleted.get('exports', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
