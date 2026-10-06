import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings
from backend.app.models import ModelGateway
from data_pipeline.law import append_public_collections, build_public_records, rebuild_public_collections


def main() -> int:
    parser = argparse.ArgumentParser(description="重建或追加公共法律知识库")
    parser.add_argument("--data-dir", default="data/public", help="公共法律原始数据目录")
    parser.add_argument("--output-dir", default="data/processed", help="处理结果输出目录")
    parser.add_argument("--write-milvus", action="store_true", help="写入 Milvus")
    parser.add_argument("--append", action="store_true", help="追加写入而不是安全替换")
    args = parser.parse_args()

    result = build_public_records(Path(args.data_dir), Path(args.output_dir))
    print({"collections": result["quality"], "output_dir": str(Path(args.output_dir))})
    if not args.write_milvus:
        return 0
    settings = get_settings()
    model = ModelGateway(settings)
    writer = append_public_collections if args.append else rebuild_public_collections
    print(writer(settings=settings, collections=result["collections"], model=model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
