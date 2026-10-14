from pathlib import Path
import os

from corpus import load_corpus


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_DIR = Path(r"C:\Users\lenovo\Desktop\RAG 工单 (1)\RAG 工单\附件\ccf_competition")


def main() -> None:
    corpus_dir = Path(os.getenv("CCF_DATA_DIR", str(DEFAULT_CORPUS_DIR)))
    chunks = load_corpus(corpus_dir)
    from retrieval import build_index

    build_index(chunks, PROJECT_DIR / "data")
    print(f"已读取9份年报，共生成 {len(chunks)} 个文本块。")
    print(f"索引已保存到：{PROJECT_DIR / 'data'}")


if __name__ == "__main__":
    main()
