"""构建知识库索引（扫描 raw 目录下的 PDF/txt/md/json，分块、向量化并写入 Milvus）"""
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，保证能 import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import logger
from src.rag.knowledge import KnowledgeService


def build_index(raw_dir: str = None, drop_existing: bool = False):
    raw_dir = raw_dir or str(PROJECT_ROOT / "data" / "raw")
    svc = KnowledgeService()
    result = svc.ingest_directory(raw_dir, drop_existing=drop_existing)

    logger.info(
        f"索引构建完成：处理 {result['files']} 个文件，共 {result['chunks']} 个分块，"
        f"失败 {len(result['errors'])} 个"
    )
    for err in result["errors"]:
        logger.error(f"  ✗ {err['file']}: {err['error']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="构建知识库索引")
    parser.add_argument("--raw_dir", default=None, help="原始文档目录（默认 data/raw）")
    parser.add_argument("--drop", action="store_true", help="先删除已有集合再重建")
    args = parser.parse_args()
    build_index(raw_dir=args.raw_dir, drop_existing=args.drop)
