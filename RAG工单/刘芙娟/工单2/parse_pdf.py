# -*- coding: utf-8 -*-
"""S2 解析：用本机 MinerU 把 PDF 解析为带页码的结构化产物。

输入   data/source_data/*.pdf
输出   data/parsed/{doc_id}/{pdf名}/{auto|txt}/{pdf名}_content_list.json
       doc_id = PDF 内容 sha256 前 12 位；每项带 page_idx（页码）
       middle.json / *.md / *.pdf 为调试产物，不入管线
配置   .mineru/mineru.json（项目内配置，不碰用户全局配置）

执行   D:/zg6_Project/9/med_rag/rag/python.exe backend/parse_pdf.py [INPUT] [选项]
       --dry-run 只看将要做什么   --force 强制重解析   --backend vlm-engine
退出码 0 成功 / 1 参数错误 / 2 校验失败 / 3 外部依赖不可用

本文件是入口薄壳：输入发现 + 编排 + 命令行。共享常量见 parse/__init__.py，
MinerU 外部依赖见 parse/mineru_env.py，产物操作见 parse/artifacts.py。
决策与偏离记录见 specs/001-mineru-pdf-parse/ 与 docs/04，此处不复述。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

# 两种调用方式都要支持：
#   `python -m backend.parse_pdf`  __package__ == "backend" → 用相对导入
#   `python backend/parse_pdf.py`  __package__ 为空，sys.path[0] 为 backend/ → 用绝对导入
# backend/ 虽无 __init__.py，但在 Python 3 是隐式命名空间包，`-m` 是能导入的，
# 所以上面的相对导入分支不是死代码，不能删。
try:
    from .parse import BACKENDS, DEFAULT_BACKEND, DEFAULT_CONFIG_PATH
    from .parse import DEFAULT_PIPELINE_MODELS, DEFAULT_VLM_MODELS
    from .parse import EXIT_ARG, EXIT_OK, PARSED_DIR, REPO_ROOT, SOURCE_DIR, ScriptError
    from .parse import artifacts, mineru_env
except ImportError:
    from parse import BACKENDS, DEFAULT_BACKEND, DEFAULT_CONFIG_PATH
    from parse import DEFAULT_PIPELINE_MODELS, DEFAULT_VLM_MODELS
    from parse import EXIT_ARG, EXIT_OK, PARSED_DIR, REPO_ROOT, SOURCE_DIR, ScriptError
    from parse import artifacts, mineru_env


# --- 输入发现 --------------------------------------------------------------- #

def compute_doc_id(pdf_path: Path) -> str:
    """doc_id = 文件内容 sha256 前 12 位；内容一变 doc_id 即变（docs/04 §10.1）。"""
    digest = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def discover_documents(input_path: Path) -> list[tuple[Path, str]]:
    """返回 [(pdf 路径, doc_id), ...]，按内容去重。"""
    if input_path.is_dir():
        candidates = sorted(p for p in input_path.iterdir() if p.suffix.lower() == ".pdf")
        if not candidates:
            raise ScriptError(EXIT_ARG, f"目录中没有 PDF 文件：{input_path}")
    elif input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ScriptError(EXIT_ARG, f"不是 PDF 文件：{input_path}")
        candidates = [input_path]
    else:
        raise ScriptError(EXIT_ARG, f"路径不存在：{input_path}")

    unique: dict[str, Path] = {}
    for path in candidates:
        doc_id = compute_doc_id(path)
        if doc_id in unique:
            print(f"WARN: {path.name} 与 {unique[doc_id].name} 内容相同，按同一文档处理，跳过")
            continue
        unique[doc_id] = path
    return [(path, doc_id) for doc_id, path in sorted(unique.items())]


# --- 单文档流程与命令行 ------------------------------------------------------ #

def process_one(pdf_path: Path, doc_id: str, args: argparse.Namespace, mineru_exe: Path,
                mineru_version: str, env: dict[str, str]) -> str:
    """处理一份文档，返回 'parsed' 或 'skipped'。"""
    doc_dir = PARSED_DIR / doc_id
    if not args.force and artifacts.is_artifact_complete(doc_dir):
        return "skipped"
    if doc_dir.exists():
        shutil.rmtree(doc_dir)  # 清理上次中途失败的残留，避免半份产物被误判为完成
    doc_dir.mkdir(parents=True, exist_ok=True)

    mineru_env.run_mineru(mineru_exe, pdf_path, doc_dir, args.backend, env)

    items = artifacts.load_content_items(artifacts.locate_content_list(doc_dir))
    artifacts.validate_page_idx(items)
    distribution = artifacts.build_type_distribution(items, args.backend)
    artifacts.print_report(distribution, mineru_version)

    if distribution["unknown_types"]:
        raise ScriptError(
            artifacts.EXIT_VALIDATION,
            f"出现 docs/04 §5 未覆盖的块类型：{'、'.join(distribution['unknown_types'])}\n"
            f"       停止并需人工决策——未知类型意味着可能有内容被无声跳过。")
    print(f"      产物: {doc_dir}")
    return "parsed"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用本机 MinerU 解析 PDF，产出带页码的结构化内容（S2 解析步骤）")
    parser.add_argument("input", nargs="?", default=str(SOURCE_DIR),
                        help=f"PDF 文件或目录（默认：{SOURCE_DIR}）")
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help=f"MinerU 后端（默认 {DEFAULT_BACKEND}；可选 {'/'.join(BACKENDS)}）")
    parser.add_argument("--force", action="store_true", help="强制重新解析，忽略已有产物")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="项目内配置文件路径")
    parser.add_argument("--pipeline-models", default=str(DEFAULT_PIPELINE_MODELS),
                        help="pipeline 权重目录")
    parser.add_argument("--vlm-models", default=str(DEFAULT_VLM_MODELS), help="vlm 权重目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要做什么，不调用 MinerU")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.backend not in BACKENDS:
            raise ScriptError(
                EXIT_ARG, f"--backend 取值非法：{args.backend}（可选：{'/'.join(BACKENDS)}）")
        pipeline_models, vlm_models = Path(args.pipeline_models), Path(args.vlm_models)
        weights = mineru_env.check_weights(args.backend, pipeline_models, vlm_models)
        documents = discover_documents(Path(args.input))

        if args.dry_run:
            print(f"仓库根 {REPO_ROOT}\n后端   {args.backend}（权重 {weights}）")
            print(f"配置   {args.config}\n产物   {PARSED_DIR}\n待处理 {len(documents)} 份")
            for pdf_path, doc_id in documents:
                print(f"  doc_id={doc_id}  {pdf_path.name}")
            print("（dry-run：未写配置、未调用 MinerU）")
            return EXIT_OK

        mineru_exe = mineru_env.resolve_mineru_exe()
        mineru_env.write_mineru_config(Path(args.config), pipeline_models, vlm_models)
        env = mineru_env.build_mineru_env(Path(args.config))
        mineru_version = mineru_env.probe_mineru_version(mineru_exe, env)

        print(f"MinerU {mineru_version} | 后端 {args.backend} | 待处理 {len(documents)} 份\n")
        failures: list[int] = []
        for index, (pdf_path, doc_id) in enumerate(documents, start=1):
            print(f"[{index}/{len(documents)}] doc_id={doc_id}  file={pdf_path.name}")
            try:
                status = process_one(pdf_path, doc_id, args, mineru_exe, mineru_version, env)
            except ScriptError as exc:
                failures.append(exc.exit_code)
                print(f"ERROR: {exc}", file=sys.stderr)
                continue
            if status == "skipped":
                print("      status=skipped (产物已存在且完整)")
            print()

        # 任一文档失败即返回该失败码，不因后续文档成功而回退为 0
        return max(failures) if failures else EXIT_OK
    except ScriptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
