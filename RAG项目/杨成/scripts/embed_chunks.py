import argparse
import json
import time
from pathlib import Path


EXPECTED_VECTOR_DIMENSION = 1024
DEFAULT_MODEL_NAME = r"D:\桌面缓存\bge-m3"


class EmbedChunksError(RuntimeError):
    pass


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise EmbedChunksError(f"JSONL 第 {line_number} 行解析失败：{exc}") from exc


def load_model(model_name, use_fp16):
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise EmbedChunksError("未找到 FlagEmbedding，请先安装 FlagEmbedding") from exc

    return BGEM3FlagModel(model_name, use_fp16=use_fp16)


def dense_vectors_from_output(output):
    if isinstance(output, dict):
        if "dense_vecs" not in output:
            raise EmbedChunksError("BGE-m3 输出中缺少 dense_vecs")
        vectors = output["dense_vecs"]
    else:
        vectors = output

    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    return vectors


def normalize_vector(vector):
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def embed_batch(model, batch, batch_size, max_length):
    texts = [record["text"] for record in batch]
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vectors = dense_vectors_from_output(output)
    if len(vectors) != len(batch):
        raise EmbedChunksError(f"向量数量与输入数量不一致：{len(vectors)} != {len(batch)}")

    embedded = []
    for record, vector in zip(batch, vectors):
        vector = normalize_vector(vector)
        if len(vector) != EXPECTED_VECTOR_DIMENSION:
            raise EmbedChunksError(
                f"{record.get('child_id', '<unknown>')} 向量维度异常：{len(vector)}，预期 {EXPECTED_VECTOR_DIMENSION}"
            )
        output_record = dict(record)
        output_record["vector"] = vector
        embedded.append(output_record)
    return embedded


def output_path_for(chunks_path, output_dir):
    return output_dir / f"{chunks_path.stem}_with_vectors.jsonl"


def embed_chunks(chunks_path, output_dir, model_name, batch_size, max_length, use_fp16):
    if not chunks_path.exists():
        raise EmbedChunksError(f"chunks.jsonl 不存在：{chunks_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = output_path_for(chunks_path, output_dir)
    model = load_model(model_name, use_fp16)

    total_count = 0
    success_count = 0
    skipped_count = 0
    pending = []

    with vectors_path.open("w", encoding="utf-8") as output_file:
        for record in iter_jsonl(chunks_path):
            total_count += 1
            text = record.get("text", "")
            if record.get("needs_split") is True:
                skipped_count += 1
                print(f"SKIP needs_split child_id={record.get('child_id')} length={len(text)}")
                continue
            if not text:
                raise EmbedChunksError(f"{record.get('child_id', '<unknown>')} 缺少 text 字段")

            pending.append(record)
            if len(pending) >= batch_size:
                for embedded_record in embed_batch(model, pending, batch_size, max_length):
                    output_file.write(json.dumps(embedded_record, ensure_ascii=False) + "\n")
                    success_count += 1
                pending.clear()

        if pending:
            for embedded_record in embed_batch(model, pending, batch_size, max_length):
                output_file.write(json.dumps(embedded_record, ensure_ascii=False) + "\n")
                success_count += 1

    return vectors_path, total_count, success_count, skipped_count


def should_use_fp16():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def main():
    parser = argparse.ArgumentParser(description="Embed parent-child child chunks with BGE-m3 dense vectors.")
    parser.add_argument("chunks_jsonl", help="parent_child_chunks.jsonl 文件路径")
    parser.add_argument("output_dir", help="向量化结果输出目录")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--fp16", action="store_true", help="强制使用 fp16")
    parser.add_argument("--no-fp16", action="store_true", help="强制关闭 fp16")
    args = parser.parse_args()

    use_fp16 = should_use_fp16()
    if args.fp16:
        use_fp16 = True
    if args.no_fp16:
        use_fp16 = False

    start = time.perf_counter()
    try:
        vectors_path, total_count, success_count, skipped_count = embed_chunks(
            chunks_path=Path(args.chunks_jsonl),
            output_dir=Path(args.output_dir),
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_fp16=use_fp16,
        )
    except EmbedChunksError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "input": Path(args.chunks_jsonl).as_posix(),
                "output": vectors_path.as_posix(),
                "total_count": total_count,
                "success_count": success_count,
                "skipped_count": skipped_count,
                "vector_dimension": EXPECTED_VECTOR_DIMENSION,
                "elapsed_seconds": round(elapsed, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
