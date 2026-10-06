# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/image_parser/image_parser.py —— 工单四图像多模态解析统一入口（新增文件）

职责（Step 1 工单要求）：
  1. Chinese-CLIP 图像嵌入（模型缺失时优雅跳过，输出 embedding 字段）；
  2. Qwen-VL 生成中文描述 caption（复用 image_captioner.Qwen2VLEngine）；
  3. PaddleOCR 提取图中文字（复用 image_ocr，VLM 转录兜底）；
  4. 图表类 VQA：预置问题（图中类别/数值/增长率最快/负增长）；
  5. 输出 data/image_descriptions/{doc_name}_images_parsed.json。

命令行（工单验收命令）：
  python -m src.image_parser.image_parser \
      --images "data/images/招股说明书2_images.json" \
      --out "data/image_descriptions/招股说明书2_images_parsed.json"
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from PIL import Image

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

# 工单四：Chinese-CLIP 本地模型路径（Step 4 检索嵌入复用同一引擎）
DEFAULT_CLIP_DIR = "/home/dabaie/models/chinese-clip-vit-base-patch16"


class CLIPEngine:
    """工单四：Chinese-CLIP 图像嵌入引擎（惰性加载，缺失优雅降级）"""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or DEFAULT_CLIP_DIR
        self._model = None
        self._proc = None
        self.available = False
        self._load_failed = False   # 工单四：加载失败短路，避免下载中途每问重复扫目录报错

    def _ensure_loaded(self) -> bool:
        """工单四：加载 Chinese-CLIP；模型不存在/不完整返回 False（容错规范）"""
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        if not Path(self.model_dir).exists():
            logger.warning(f"[clip] 模型缺失 {self.model_dir}，跳过图像嵌入")
            self._load_failed = True
            return False
        try:
            import torch
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
            self._proc = ChineseCLIPProcessor.from_pretrained(self.model_dir)
            self._model = ChineseCLIPModel.from_pretrained(
                self.model_dir, torch_dtype=torch.float16,
                device_map="cuda" if torch.cuda.is_available() else "cpu")
            self._model.eval()
            self.available = True
            logger.info("[clip] Chinese-CLIP 就绪")
            return True
        except Exception as e:                      # 工单四：加载失败不阻塞解析
            logger.warning(f"[clip] 加载失败，跳过嵌入: {e}")
            self._load_failed = True
            return False

    def embed_image(self, image: Image.Image) -> Optional[List[float]]:
        """工单四：单图嵌入（512 维）；不可用返回 None"""
        if not self._ensure_loaded():
            return None
        import torch
        inputs = self._proc(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self._model.get_image_features(**inputs)
        # 工单四：兼容 transformers>=5.x（返回 BaseModelOutputWithPooling，
        # 投影特征在 pooler_output）与旧版（直接返回特征 tensor）
        feats = getattr(out, "pooler_output", None)
        if feats is None:
            feats = out
        feats = feats / feats.norm(dim=-1, keepdim=True)   # 工单四：L2 归一化
        return feats[0].float().cpu().tolist()


class ImageParser:
    """工单四：图像多模态解析编排（嵌入 + 描述 + OCR + VQA）"""

    def __init__(self, vlm_engine: Optional[Any] = None,
                 clip_engine: Optional[CLIPEngine] = None,
                 enable_vqa: bool = True, enable_ocr: bool = True,
                 enable_clip: bool = True):
        # 工单四：复用既有 captioner 流水线（caption+VQA+OCR 三合一）
        from src.image_parser.image_captioner import ImageCaptioner
        self._cap = ImageCaptioner(vlm_engine=vlm_engine,
                                   enable_vqa=enable_vqa, enable_ocr=enable_ocr)
        self.clip = clip_engine or (CLIPEngine() if enable_clip else None)

    # ------------------------------------------------------------------
    def parse_one(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """工单四：解析单图 = caption/VQA/OCR（复用） + CLIP 嵌入（新增）"""
        out = self._cap.parse_one(meta)
        out["embedding"] = None                     # 工单四：CLIP 嵌入默认空
        if self.clip is not None and out.get("parse_status") != "failed":
            try:
                with Image.open(meta["path"]) as img:
                    out["embedding"] = self.clip.embed_image(img)
            except Exception as e:                  # 工单四：嵌入失败不阻塞
                logger.warning(f"[parser] {meta.get('image_id')} 嵌入失败: {e}")
        return out

    # ------------------------------------------------------------------
    def parse_images(self, images_json: str, out_json: str,
                     limit: Optional[int] = None,
                     image_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """工单四：整册解析 → data/image_descriptions/{doc_name}_images_parsed.json"""
        src = json.loads(Path(images_json).read_text(encoding="utf-8"))
        metas = src.get("images", [])
        if image_ids:
            metas = [m for m in metas if m.get("image_id") in image_ids]
        if limit:
            metas = metas[:limit]

        parsed = []
        for i, meta in enumerate(metas, 1):
            logger.info(f"[parser] ({i}/{len(metas)}) {meta.get('image_id')} p{meta.get('page')}")
            parsed.append(self.parse_one(meta))

        emb_cnt = sum(1 for p in parsed if p.get("embedding"))
        result = {
            "doc_id": src.get("doc_id"),
            "source_manifest": images_json,
            "work_order": WORK_ORDER,               # 工单四：工单编号溯源
            "vlm_engine": getattr(self._cap.vlm, "engine_name", "mock"),
            "clip_enabled": bool(emb_cnt),
            "images": parsed,
            "stats": {
                "total": len(parsed),
                "ok": sum(1 for p in parsed if p["parse_status"] == "ok"),
                "with_embedding": emb_cnt,
            },
        }
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        logger.info(f"[parser] 完成 -> {out_path}")
        return result


def main() -> None:
    """工单四：命令行入口（验收命令签名）"""
    parser = argparse.ArgumentParser(
        description="工单四 图像多模态解析（人工智能NLP-RAG-图像内容解析及检索优化）")
    parser.add_argument("--images", required=True, help="提取清单 JSON")
    parser.add_argument("--out", required=True, help="解析结果 JSON")
    parser.add_argument("--model-dir", default=None, help="Qwen2-VL 模型目录")
    parser.add_argument("--clip-dir", default=None, help="Chinese-CLIP 模型目录")
    parser.add_argument("--limit", type=int, default=None, help="仅解析前 N 张")
    parser.add_argument("--image-ids", nargs="*", default=None, help="指定 image_id")
    parser.add_argument("--skip-vqa", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    args = parser.parse_args()

    from src.image_parser.image_captioner import Qwen2VLEngine
    vlm = Qwen2VLEngine(model_dir=args.model_dir) if args.model_dir else Qwen2VLEngine()
    clip = CLIPEngine(model_dir=args.clip_dir) if args.clip_dir else (
        None if args.skip_clip else CLIPEngine())
    ip = ImageParser(vlm_engine=vlm, clip_engine=clip,
                     enable_vqa=not args.skip_vqa, enable_ocr=not args.skip_ocr)
    result = ip.parse_images(args.images, args.out,
                             limit=args.limit, image_ids=args.image_ids)

    # 工单四：验收摘要输出
    s = result["stats"]
    print(f"\n=== 图像解析完成 ===")
    print(f"总计 {s['total']} 张 | 成功 {s['ok']} | CLIP嵌入 {s['with_embedding']}")
    for img in result["images"][:3]:
        print(f"\n[{img['image_id']}] p{img['page']} {img.get('parse_status')}")
        print(f"  caption: {img.get('caption', '')[:120]}")
        print(f"  ocr: {img.get('ocr_text', '')[:80]}")
        for qa in img.get("vqa_qa", [])[:2]:
            print(f"  VQA-Q: {qa['q']}")
            print(f"  VQA-A: {qa['a'][:100]}")


if __name__ == "__main__":
    main()
