"""
图像语义解析脚本（工单4 步骤 2/2）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

两条路线都在这里落地（工单备注写的是「使用多模态模型（**CLIP 或**多模态大模型）实现」）：

  ① 多模态大模型（默认）：对每张图调 qwen-vl-plus 生成**结构化文本描述**，
     回填 desc —— 检索时命中的是这段文本。这是 id=5 / id=6 能答对的直接原因。
  ② CLIP（--clip）：用 Chinese-CLIP 算出每张图的**跨模态向量**，
     落盘 clip_embeddings.npy，并回填每个图的 clip_index ——
     让「以文搜图」直接成立（问题向量与图向量在同一空间里比相似度）。

两条路线解决的不是同一件事，所以都保留：
描述解决「图里的内容能不能被文字检索命中」，向量解决「措辞对不上时的兜底召回」。

已解析过的图会跳过，支持断点续跑（VL 调用单张 3~8s，68 张要几分钟）。

用法：
    python scripts/parse_images.py                     # 全部（多模态大模型）
    python scripts/parse_images.py --clip-only         # 只算 CLIP 图向量
    python scripts/parse_images.py --clip              # 解析 + 算 CLIP 向量
    python scripts/parse_images.py --doc liyuan --page 39 39 72   # 只跑指定页（调试）
    python scripts/parse_images.py --no-verify         # 关掉第二遍校验（对照实验）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.image_semantics import describe_image, render_description, save_figures  # noqa: E402


def build_clip(figures: list[dict]) -> int:
    """算 CLIP 图向量并落盘。返回编码成功的图数。

    失败（模型没下全 / 依赖缺失）只打印原因并返回 0 —— CLIP 是**可选增强通道**，
    它挂了不该拦住整个解析流程，检索端会自动降级回纯文本。
    """
    from src import clip_encoder

    enc = clip_encoder.ClipEncoder.instance()
    if not enc.available:
        print(f"[CLIP] 跳过：模型目录不存在 {enc.model_path}")
        return 0
    if not enc.load():
        print(f"[CLIP] 跳过：模型加载失败 {enc.load_error}")
        return 0
    t0 = time.time()
    info = clip_encoder.build_and_save(figures)
    if not info.get("count"):
        print(f"[CLIP] 跳过：{info.get('error')}")
        return 0
    print(f"[CLIP] {info['count']} 张图 → {info['path']}（{info['dim']} 维，"
          f"模型 {info['model']}，耗时 {time.time() - t0:.1f}s）")
    return int(info["count"])


def main() -> int:
    ap = argparse.ArgumentParser(description="图像语义解析 · 多模态大模型 + CLIP（工单4）")
    ap.add_argument("--doc", default="", help="只处理某份文档（liyuan/xingtu）")
    ap.add_argument("--page", nargs="*", type=int, default=None, help="只处理这些页")
    ap.add_argument("--no-verify", action="store_true", help="关闭第二遍校验（消融用）")
    ap.add_argument("--reset", action="store_true", help="清掉已有解析结果重跑")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（调试）")
    ap.add_argument("--clip", action="store_true", help="解析后顺带算 CLIP 图向量")
    ap.add_argument("--clip-only", action="store_true", help="只算 CLIP 图向量，不调大模型")
    ap.add_argument("--rerender-only", action="store_true",
                    help="不调模型，只用已有 semantics + 几何拓扑重算 desc"
                         "（改了描述模板/拓扑后用它，省下 8 分钟和调用费）")
    args = ap.parse_args()

    config.ensure_dirs()
    figures = json.loads(config.FIGURES_JSON.read_text(encoding="utf-8"))

    if args.rerender_only:
        from src.image_semantics import ImageSemantics, render_description

        n = 0
        for f in figures:
            if not f.get("semantics"):
                continue
            sem = ImageSemantics(**{k: v for k, v in f["semantics"].items()
                                    if k in ImageSemantics.__dataclass_fields__})
            f["desc"] = render_description(sem, f)
            n += 1
        save_figures(figures)
        print(f"已用几何拓扑重算 {n} 张图的描述 → {config.FIGURES_JSON.name}")
        return 0

    if args.clip_only:
        build_clip(figures)
        save_figures(figures)   # 回填 clip_index
        return 0

    if args.reset:
        for f in figures:
            f.pop("desc", None)
            f.pop("semantics", None)
            f.pop("error", None)

    todo = []
    for f in figures:
        if args.doc and f["doc_key"] != args.doc:
            continue
        if args.page and f["page"] not in set(args.page):
            continue
        if f.get("desc") and not f.get("error"):
            continue
        todo.append(f)
    if args.limit:
        todo = todo[:args.limit]

    print(f"待解析 {len(todo)} / 共 {len(figures)} 张图；模型 = {config.VL_MODEL}；"
          f"二次校验 = {'关' if args.no_verify else '开'}")

    t0 = time.time()
    ok = err = verified = 0
    for i, fig in enumerate(todo, 1):
        img = config.ROOT_DIR / fig["image"]
        doc = config.DOCS_BY_KEY.get(fig["doc_key"], {})
        sem = describe_image(img, fig["page"], doc.get("name", fig["doc_key"]),
                             caption=fig.get("caption", ""), doc_key=fig["doc_key"],
                             index=fig["index"], verify=not args.no_verify)
        fig["semantics"] = sem.as_dict()
        fig["error"] = sem.error
        fig["desc"] = render_description(sem, fig) if sem.ok else ""
        fig["elapsed_ms"] = sem.elapsed_ms
        if sem.ok:
            ok += 1
        else:
            err += 1
        if sem.verify_kind:
            verified += 1
        flag = "OK " if sem.ok else "ERR"
        vk = f"+{sem.verify_kind}" if sem.verify_kind else "   "
        print(f"  [{i:>3}/{len(todo)}] {flag} {vk} {fig['doc_key']} p{fig['page']:>3}#{fig['index']} "
              f"{sem.elapsed_ms:>5}ms {sem.fig_type[:14]}")
        if not sem.ok:
            print(f"        {sem.error[:150]}")

    # 全部图（含未参与本轮解析的）一起落盘，保证清单完整
    if not args.doc and not args.page and not args.limit:
        save_figures(figures)
    else:
        save_figures(figures)

    print(f"\n完成：成功 {ok}，失败 {err}，其中有二次校验 {verified}；"
          f"耗时 {time.time() - t0:.1f}s")
    print(f"清单 → {config.FIGURES_JSON.relative_to(config.ROOT_DIR)}")

    if args.clip:
        build_clip(figures)
        save_figures(figures)   # 回填 clip_index
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
