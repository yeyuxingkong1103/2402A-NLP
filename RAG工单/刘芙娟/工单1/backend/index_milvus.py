#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""S6 入库：把 S4/S5 的产物写入 Milvus。

（本 docstring 是 raw 字符串：里面的 Windows 路径含 \z \r \i 等片段，
  不加 r 前缀会触发 SyntaxWarning，且 \r 会被解释成回车。）

============================ 这个脚本做什么 ============================

输入：  data/chunks/{doc_id}.chunks.jsonl          分块产物（S4）
        data/embeddings/{doc_id}.npy               float32[N, 1024]，L2 归一化（S5）
        data/embeddings/{doc_id}.rows.jsonl        .npy 的行 ↔ chunk_id 对齐表
        data/embeddings/{doc_id}.fingerprint.json  模型指纹

输出：  Milvus collection med_rag_v1                 一行一个 chunk
        data\index_manifest.json                   索引的「身份证」，后端启动时校验
        data\index_backup\{doc_id}.rollback.jsonl  仅在需要删旧数据时写（回滚依据）

        三处路径均可用命令行参数覆盖（--manifest / --backup-dir 等）。

实现拆在 backend/index/ 下（inputs 读产物与校验 / versions 参数版本 / store **唯一会写库
的模块** / manifest 索引清单 / report 报告），本文件是唯一入口。

============================ 运行方法 ============================

  先装依赖（只需一次）：
    D:/zg6_Project/9/med_rag/rag/python.exe -m pip install pymilvus

  只读自检（不写库、不写清单）：
    D:/zg6_Project/9/med_rag/rag/python.exe backend/index_milvus.py --print-env
    D:/zg6_Project/9/med_rag/rag/python.exe backend/index_milvus.py --print-config

  入库（文档标识留空 = 处理 data/chunks/ 下全部文档）：
    D:/zg6_Project/9/med_rag/rag/python.exe backend/index_milvus.py
    D:/zg6_Project/9/med_rag/rag/python.exe backend/index_milvus.py d6da41b5d356
    D:/zg6_Project/9/med_rag/rag/python.exe backend/index_milvus.py --dry-run    只校验、判门禁

  退出码：0 成功 ／ 1 参数错误 ／ 2 校验失败（或写入失败但**已回滚**）／
          3 Milvus 不可用（含 pymilvus 未安装）／ 4 参数版本不符，需人工决定是否 --rebuild ／
          5 回滚也失败 —— **库状态不确定，需人工介入**

============================ 三条必须知道的约定 ============================

1. **幂等键是 `doc_id`，不是 `source_hash`。** `doc_id` 是源 PDF 的 sha256 前 12 位，由文件
   内容决定，所以同一份 PDF 重跑整条管线天然幂等。而产物里的 `source_hash` 是 MinerU
   `_origin.pdf` 副本的哈希 —— 换了 MinerU 版本重跑解析它就会变，用它做删除键会让新旧
   版本并存、检索命中同一段落的两个版本。`source_hash` 只照实落库，不参与删除（D1 裁决）。

2. **参数版本不符时拒绝写入，不自动重建。** `pipeline_config_hash` 覆盖分块数值参数、清洗/
   分块规则版本、embedding 模型指纹。不一致就报退出码 4 并逐项列出差异 —— 自动重建会在
   无人知晓时清空索引，而它一旦失败，系统就从「可用但参数旧」变成「完全不可用」。
   要重建必须显式 `--rebuild`。

3. **索引建在空集合上。** `docs/04 §9.2` 要求「写入完成后统一建索引」，但 Milvus 上做不到：
   delete/query 要求集合已 load，而 load 要求索引已存在。索引因此随建表一起创建，此时集合
   是空的 —— §9.2 想避免的「索引看到写入的中途状态」因此天然不会发生。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from index import (  # noqa: E402
    COLLECTION_NAME,
    DEFAULT_URI,
    DIM,
    EXIT_ARGS,
    EXIT_DEP,
    EXIT_GATE,
    EXIT_OK,
    EXIT_ROLLBACK,
    EXIT_VALIDATION,
    INDEX_TYPE,
    METRIC_TYPE,
    SEVERITY,
    TOKEN_ENV,
    IngestError,
)
from index import inputs, manifest as manifest_mod, report, store, versions  # noqa: E402

# 查询期参数，**不参与** pipeline_config_hash（docs/04 §10.2 末段）。
# 阈值 0.6 是当前取值；项目文档（docs/02 §7 说明）要求它最终由标定产生，见 plan.md「已确定 2」。
APP_CONFIG = {"top_k": 3, "similarity_threshold": 0.6}

# 输入产物与入库产出都在项目内的 data\ 下（与 docs/04 §9.3 一致）
DATA_DIR = os.path.join(REPO_ROOT, "data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="index_milvus.py",
        description="S6 入库：把 S4/S5 的产物写入 Milvus，并写出 data/index_manifest.json",
    )
    parser.add_argument("doc_ids", nargs="*", help="文档标识；留空 = 处理 data/chunks/ 下全部文档")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Milvus 地址，默认 %s" % DEFAULT_URI)
    parser.add_argument("--chunks-dir", default=os.path.join(DATA_DIR, "chunks"))
    parser.add_argument("--emb-dir", default=os.path.join(DATA_DIR, "embeddings"))
    parser.add_argument("--manifest", default=os.path.join(DATA_DIR, "index_manifest.json"))
    parser.add_argument("--backup-dir", default=os.path.join(DATA_DIR, "index_backup"))
    parser.add_argument("--rebuild", action="store_true", help="破坏性：忽略版本门禁后重建")
    parser.add_argument("--dry-run", action="store_true", help="只校验与判门禁，不写库、不写清单")
    parser.add_argument("--print-config", action="store_true", help="打印本次参数清单与哈希后退出")
    parser.add_argument("--print-env", action="store_true", help="打印运行环境后退出")
    return parser


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_all(doc_ids: list[str], args: argparse.Namespace) -> list:
    """先把**全部**文档读入并校验完，再连库 —— FR-005：一切校验先于任何写操作。"""
    return [inputs.load_doc(doc_id, args.chunks_dir, args.emb_dir) for doc_id in doc_ids]


def assert_uniform(loaded: list) -> None:
    """多份文档必须由同一套参数产出，否则算出的 hash 代表不了任何一份。"""
    rules = {doc.chunk_rule_version for doc in loaded}
    if len(rules) != 1:
        raise IngestError(EXIT_VALIDATION, "多份文档的 chunk_rule_version 不一致：%s" % sorted(rules))
    fingerprints = {versions.compute_hash({"embed": doc.fingerprint}) for doc in loaded}
    if len(fingerprints) != 1:
        raise IngestError(
            EXIT_VALIDATION,
            "多份文档的 embedding 指纹不一致，不能共用一份 pipeline_config_hash。"
            "请分开入库（每次只给一份文档）",
        )


def check_gate(client, config: dict, config_hash: str, old_config: dict | None,
               rebuild: bool) -> int | None:
    """返回 None 表示放行；返回退出码表示拒绝（docs/04 §10.2）。"""
    stored = store.read_stored_hash(client)
    if stored is None:
        report.step(3, "版本门禁 … 集合为空，无门禁")
        return None
    if stored == config_hash:
        report.step(3, "版本门禁 … 与库内一致（%s）" % config_hash[:16])
        return None

    diff = versions.describe_diff(old_config, config)
    if not rebuild:
        report.step(3, "版本门禁 … **不符**，拒绝写入")
        report.diff_block(diff)
        report.say("")
        report.say("  库内：%s" % stored)
        report.say("  本次：%s" % config_hash)
        report.say("  参数已变更。自动重建会在无人知晓时清空索引，因此必须人工显式发起：")
        report.say("    D:/zg6_Project/9/med_rag/rag/python.exe backend/index_milvus.py --rebuild")
        return EXIT_GATE
    report.step(3, "版本门禁 … 不符，但已显式 --rebuild，继续")
    report.diff_block(diff)
    return None


def commit_all(client, loaded: list, config_hash: str, backup_dir: str):
    """逐份原子提交。返回 (成功结果, 失败明细, 是否因回滚失败中止)。"""
    results, failures, aborted = [], [], False
    for doc in loaded:
        try:
            result = store.commit_document(client, doc, config_hash, backup_dir)
        except IngestError as exc:
            report.failure(doc.doc_id, exc.message)
            failures.append((doc.doc_id, exc.message, exc.code))
            if exc.code == EXIT_ROLLBACK:
                # 库状态不确定 —— 立即停止，不再动其它文档
                aborted = True
                break
            continue
        results.append(result)
        report.say(
            "      ✓ %s：旧 %d 行，删 %d 行，插 %d 行，校验 %d == %d"
            % (result.doc_id, result.old_count, result.deleted, result.inserted,
               result.final_count, result.expected)
        )
    return results, failures, aborted


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.print_env:
        report.env(args.uri, COLLECTION_NAME, TOKEN_ENV)
        return EXIT_OK
    if args.rebuild and args.dry_run:
        raise IngestError(EXIT_ARGS, "--rebuild 与 --dry-run 不能同时给出（一个是破坏性的，一个承诺不写）")

    doc_ids = inputs.resolve_docs(args.doc_ids, args.chunks_dir)
    loaded = load_all(doc_ids, args)
    assert_uniform(loaded)

    first = loaded[0]
    config = versions.collect_pipeline_config(first.chunk_rule_version, first.fingerprint)
    config_hash = versions.compute_hash(config)
    report.step(1, "校验产物 … %d 份文档 / %d 个 chunk，V1–V12 全部通过"
                % (len(loaded), sum(doc.count for doc in loaded)))

    if args.print_config:
        report.config_block(config, config_hash)
        return EXIT_OK

    existing = manifest_mod.read_manifest(args.manifest)
    old_config = (existing or {}).get("pipeline_config")

    client = store.connect(args.uri)
    created, index_repaired = store.ensure_collection(client)
    store.ensure_loaded(client)
    report.step(2, "连接 Milvus … %s（collection %s）"
                % (args.uri, "本次新建" if created else "已存在"))
    if index_repaired:
        report.hint("collection 已存在但没有索引，已按 V1 锁定值补建 %s/%s"
                    % (INDEX_TYPE, METRIC_TYPE))

    gate_code = check_gate(client, config, config_hash, old_config, args.rebuild)
    if gate_code is not None:
        return gate_code

    if args.dry_run:
        report.say("")
        report.say("dry-run 通过：产物校验与版本门禁均无阻塞，未写入任何数据。")
        return EXIT_OK

    report.step(4, "写入 … 逐份「取旧 → 删 → 插 → 校验」，失败即用备份回滚")
    results, failures, aborted = commit_all(client, loaded, config_hash, args.backup_dir)

    total = store.count_all(client)
    report.step(5, "核对 … 库内总行数 %d" % total)

    manifest_path = None
    if aborted:
        report.step(6, "索引清单 … 跳过（有文档回滚失败，库状态不确定，写清单只会固化错误状态）")
    else:
        built = manifest_mod.build_manifest(
            client=client, existing=existing,
            processed_doc_ids=[doc.doc_id for doc in loaded],
            config=config, config_hash=config_hash,
            app_config=APP_CONFIG, built_at=_now_iso(),
        )
        manifest_mod.write_manifest(args.manifest, built)
        manifest_path = args.manifest
        report.step(6, "索引清单 … 已更新（documents: %d，total_chunks: %d）"
                    % (len(built["documents"]), built["total_chunks"]))

    doc_lines = ["%s：提交成功（%d 行）" % (r.doc_id, r.final_count) for r in results]
    doc_lines += ["%s：失败（已停止）" % doc_id for doc_id, _, _ in failures]
    report.summary(
        collection=COLLECTION_NAME, uri=args.uri, created=created,
        index_type=INDEX_TYPE, metric_type=METRIC_TYPE, dim=DIM,
        written=sum(r.final_count for r in results), total=total,
        config_hash=config_hash, doc_lines=doc_lines, manifest_path=manifest_path,
    )

    if not failures:
        return EXIT_OK
    # 多份文档时取**最严重**的一份（SEVERITY 从重到轻）
    return min((code for _, _, code in failures), key=SEVERITY.index)


def run() -> int:
    try:
        return main(sys.argv[1:])
    except IngestError as exc:
        report.say("")
        report.say("错误：%s" % exc.message)
        return exc.code
    except KeyboardInterrupt:
        report.say("")
        report.say("已中断。写入过程中的中断会触发回滚；若仍不确定库状态，"
                   "请查 data/index_backup/ 后重跑（本步骤幂等）。")
        return EXIT_DEP


if __name__ == "__main__":
    sys.exit(run())
