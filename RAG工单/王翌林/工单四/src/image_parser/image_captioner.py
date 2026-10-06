# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
src/image_parser/image_captioner.py —— 工单四多模态图像描述模块（含 VLM 引擎）

职责（见 docs/01_图像解析方案.md §5.2）：
  1. Qwen2VLEngine：Qwen2-VL-2B-Instruct 常驻引擎（fp16/GPU，batch=1 防OOM），
     提供 caption / generate / transcribe 三个原子能力；
  2. ImageCaptioner：编排 解析流水线 caption → VQA → OCR，
     合并输出 data/image_descriptions/{doc_name}_images_parsed.json。

命令行（工单验收命令）：
  python -m src.image_parser.image_captioner \
      --images "data/images/招股说明书2_images.json" \
      --out "data/image_descriptions/招股说明书2_images_parsed.json"
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from PIL import Image

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

# 工单四：本地模型路径（模型统一存放 /home/dabaie/models，可用环境变量覆盖）
DEFAULT_MODEL_DIR = "/home/dabaie/models/Qwen2-VL-2B-Instruct"
CAPTION_PROMPT = "用一段简洁的中文描述这张图：图的类型、主题内容与关键信息，60到150字。"


class Qwen2VLEngine:
    """工单四：Qwen2-VL 常驻推理引擎（惰性加载，单例复用）"""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self._model = None
        self._processor = None
        self.engine_name = "qwen2vl"

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """工单四：惰性加载模型与预处理器（fp16 + GPU；OOM 自动降级 CPU）"""
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        if not Path(self.model_dir).exists():
            raise FileNotFoundError(
                f"Qwen2-VL 模型不存在: {self.model_dir}，请先运行 "
                f"scripts/download_qwen2vl_step3.sh")
        logger.info(f"[qwen2vl] 加载模型: {self.model_dir}")
        self._processor = AutoProcessor.from_pretrained(
            self.model_dir, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
        try:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_dir, torch_dtype=torch.float16,
                device_map="cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:                     # 工单四：显存不足降级 CPU（容错规范）
            logger.warning(f"[qwen2vl] GPU 加载失败({e})，降级 CPU")
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_dir, torch_dtype=torch.float32, device_map="cpu")
        self._model.eval()
        logger.info("[qwen2vl] 模型就绪")

    # ------------------------------------------------------------------
    def generate(self, image: Image.Image, prompt: str,
                 max_new_tokens: int = 512) -> str:
        """工单四：单图单问推理（chat template + 图像消息）"""
        self._ensure_loaded()
        import torch

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text], images=[image.convert("RGB")], return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens,
                                       do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]  # 工单四：截掉 prompt 部分
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True)[0].strip()

    # ------------------------------------------------------------------
    def caption(self, image: Image.Image) -> str:
        """工单四：生成中文图像描述"""
        return self.generate(image, CAPTION_PROMPT, max_new_tokens=320)

    def transcribe(self, image: Image.Image) -> str:
        """工单四：图内文字转录（OCR 兜底通道）"""
        from src.image_parser.image_ocr import transcribe_prompt
        return self.generate(image, transcribe_prompt(), max_new_tokens=512)


class ImageCaptioner:
    """工单四：图像解析编排器（caption + VQA + OCR → 合并 JSON）"""

    def __init__(self, vlm_engine: Optional[Any] = None,
                 enable_vqa: bool = True, enable_ocr: bool = True):
        # 工单四：支持注入 Mock 引擎（单测），缺省用 Qwen2VL
        self.vlm = vlm_engine or Qwen2VLEngine()
        self.enable_vqa = enable_vqa
        self.enable_ocr = enable_ocr
        # 工单四：VQA 与 OCR 引擎延迟构造（依赖 vlm 能力）
        self._vqa = None
        self._ocr = None

    # ------------------------------------------------------------------
    def _get_vqa(self):
        if self._vqa is None:
            from src.image_parser.image_vqa import ImageVQAEngine
            self._vqa = ImageVQAEngine(self.vlm)
        return self._vqa

    def _get_ocr(self):
        if self._ocr is None:
            from src.image_parser.image_ocr import ImageOCREngine
            transcribe = getattr(self.vlm, "transcribe", None)
            self._ocr = ImageOCREngine(vlm_transcribe_fn=transcribe)
        return self._ocr

    # ------------------------------------------------------------------
    def parse_one(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """工单四：解析单张图像 → caption/ocr_text/vqa_qa，失败不阻塞"""
        out = dict(meta)
        status = []
        try:
            img = Image.open(meta["path"])
            # 1) 工单四：中文描述
            try:
                out["caption"] = self.vlm.caption(img)
                status.append("caption_ok")
            except Exception as e:
                logger.warning(f"[captioner] {meta.get('image_id')} caption 失败: {e}")
                out["caption"] = ""
                status.append("caption_failed")
            # 2) 工单四：VQA（图表/结构图按模板提问）
            if self.enable_vqa:
                try:
                    # 工单四：把已生成的 caption 传入模板判定，
                    # 否则 path 无"组织结构图/柱状图"关键词时会退化为通用问题
                    out["vqa_qa"] = self._get_vqa().run(
                        img, out, caption=out.get("caption", ""))
                    status.append("vqa_ok")
                except Exception as e:
                    logger.warning(f"[captioner] {meta.get('image_id')} vqa 失败: {e}")
                    out["vqa_qa"] = []
                    status.append("vqa_failed")
            else:
                out["vqa_qa"] = []
            # 3) 工单四：OCR（PaddleOCR 优先，VLM 转录兜底）
            if self.enable_ocr:
                ocr_res = self._get_ocr().recognize(img)
                out["ocr_text"] = ocr_res["ocr_text"]
                out["ocr_engine"] = ocr_res["engine"]
                status.append(f"ocr_{ocr_res['status']}")
            else:
                out["ocr_text"] = ""
        except Exception as e:                          # 工单四：图像文件级失败
            logger.error(f"[captioner] {meta.get('image_id')} 解析异常: {e}")
            out.setdefault("caption", "")
            out.setdefault("vqa_qa", [])
            out.setdefault("ocr_text", "")
            status.append("file_failed")
        out["parse_status"] = "failed" if "file_failed" in status or "caption_failed" in status else "ok"
        out["parse_steps"] = status
        return out

    # ------------------------------------------------------------------
    def parse_images(self, images_json: str, out_json: str,
                     limit: Optional[int] = None,
                     image_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """工单四：解析整册清单 JSON → 合并输出 parsed JSON"""
        src = json.loads(Path(images_json).read_text(encoding="utf-8"))
        metas = src.get("images", [])
        if image_ids:                                   # 工单四：支持指定 image_id 子集
            metas = [m for m in metas if m.get("image_id") in image_ids]
        if limit:
            metas = metas[:limit]

        t0 = time.time()
        parsed: List[Dict[str, Any]] = []
        for i, meta in enumerate(metas, 1):
            logger.info(f"[captioner] ({i}/{len(metas)}) {meta.get('image_id')} "
                        f"p{meta.get('page')} 解析中...")
            parsed.append(self.parse_one(meta))

        ok = sum(1 for p in parsed if p["parse_status"] == "ok")
        result = {
            "doc_id": src.get("doc_id"),
            "source_manifest": images_json,
            "work_order": WORK_ORDER,                   # 工单四：工单编号溯源
            "vlm_engine": getattr(self.vlm, "engine_name", "mock"),
            "images": parsed,
            "stats": {
                "total": len(parsed), "ok": ok,
                "failed": len(parsed) - ok,
                "elapsed_sec": round(time.time() - t0, 1),
            },
        }
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        logger.info(f"[captioner] 完成 {len(parsed)} 张 "
                    f"(ok={ok}) -> {out_path} 耗时 {result['stats']['elapsed_sec']}s")
        return result


def main() -> None:
    """工单四：命令行入口（caption + vqa + ocr 一站式合并）"""
    parser = argparse.ArgumentParser(
        description="工单四 多模态图像解析（人工智能NLP-RAG-图像内容解析及检索优化）")
    parser.add_argument("--images", required=True, help="提取清单 JSON（Step 2 产出）")
    parser.add_argument("--out", required=True, help="解析结果 JSON 输出路径")
    parser.add_argument("--model-dir", default=None, help="Qwen2-VL 模型目录")
    parser.add_argument("--limit", type=int, default=None, help="仅解析前 N 张（冒烟）")
    parser.add_argument("--image-ids", nargs="*", default=None,
                        help="仅解析指定 image_id")
    parser.add_argument("--skip-vqa", action="store_true", help="跳过 VQA")
    parser.add_argument("--skip-ocr", action="store_true", help="跳过 OCR")
    args = parser.parse_args()

    engine = Qwen2VLEngine(model_dir=args.model_dir) if args.model_dir else Qwen2VLEngine()
    captioner = ImageCaptioner(vlm_engine=engine,
                               enable_vqa=not args.skip_vqa,
                               enable_ocr=not args.skip_ocr)
    result = captioner.parse_images(args.images, args.out,
                                    limit=args.limit, image_ids=args.image_ids)
    # 工单四：终端摘要（验收输出）
    print(f"\n=== 解析完成 ===")
    print(f"总计 {result['stats']['total']} 张，成功 {result['stats']['ok']}，"
          f"耗时 {result['stats']['elapsed_sec']}s -> {args.out}")
    for img in result["images"][:3]:
        print(f"\n[{img['image_id']}] p{img['page']}")
        print(f"  caption: {img['caption'][:120]}")
        print(f"  ocr: {img['ocr_text'][:80]}")


if __name__ == "__main__":
    main()
