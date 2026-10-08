# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
tests/test_image_parser.py —— 工单四多模态图像解析单测

策略：
  - Mock VLM 引擎（不加载真实模型），验证编排/合并/降级/问题模板选择逻辑；
  - 真实 Qwen2-VL 集成测试由环境变量 RUN_VLM_TESTS=1 显式开启（模型大，避免误加载）。
"""
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from src.image_parser.image_captioner import ImageCaptioner, Qwen2VLEngine
from src.image_parser.image_ocr import ImageOCREngine


@pytest.fixture(autouse=True)
def _no_real_paddle(monkeypatch):
    """工单四：单测隔离真实 PaddleOCR（本文件验证降级/兜底逻辑，非 OCR 集成）；
    真实 PaddleOCR 已在本环境可用，其集成验证由 Step 3 重点图解析承担"""
    monkeypatch.setattr(ImageOCREngine, "_get_paddle", lambda self: None)


from src.image_parser.image_vqa import (CHART_QUESTIONS, ORG_QUESTIONS,
                                        ImageVQAEngine)

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"
MODEL_DIR = Path("/home/dabaie/models/Qwen2-VL-2B-Instruct")


# ----------------------------------------------------------------------
class MockVLM:
    """工单四：Mock VLM 引擎——caption/transcribe/generate 返回固定文本"""

    engine_name = "mock"

    def caption(self, image):
        return "组织结构图：展示某公司总经理下设销售部、研发部等部门的层级关系。"

    def transcribe(self, image):
        return "总经理 销售部 大客户销售部 销售处"

    def generate(self, image, prompt, max_new_tokens=512):
        if "增长率最快" in prompt:
            return "汽车，14.0%"
        if "负增长" in prompt:
            return "IC卡，-2.0%"
        if "数值" in prompt:
            return "计算机 42%，消费 26%，网络通信 20%"
        if "哪些类别" in prompt:
            return "计算机、消费、网络通信、工控、其他、汽车、IC卡"
        if "父子关系" in prompt or "下属" in prompt:
            return "总经理→销售部；销售部→大客户销售部；大客户销售部→6个销售处"
        if "根节点" in prompt:
            return "股东大会"
        if "流程" in prompt:
            return "客户索取→登记→免费寄送"
        return "图中内容 mock"


# ----------------------------------------------------------------------
@pytest.fixture()
def tiny_png(tmp_path):
    """工单四：生成一张小测试图"""
    p = tmp_path / "img" / "page_001_img_1.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 120), "white").save(p)
    return p


@pytest.fixture()
def manifest_json(tmp_path, tiny_png):
    """工单四：构造 Step 2 提取清单样例"""
    manifest = {
        "doc_id": "测试册",
        "work_order": WORK_ORDER,
        "images": [{
            "image_id": "img_001", "doc_id": "测试册", "page": 1,
            "image_index": 1, "bbox": [0, 0, 100, 60],
            "width": 200, "height": 120, "format": "png",
            "path": str(tiny_png), "md5": "x", "status": "ok",
            "extract_source": "L2_vector",
        }],
        "stats": {"extracted": 1},
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return p


# ----------------------------------------------------------------------
def test_ocr_engine_graceful_degradation(tiny_png):
    """工单四：OCR 引擎不可用时优雅降级（不抛异常）"""
    eng = ImageOCREngine(vlm_transcribe_fn=None)   # 无 PaddleOCR 无 VLM
    res = eng.recognize(Image.open(tiny_png))
    assert res["ocr_text"] == ""
    assert res["engine"] == "none"
    assert res["status"] in ("unavailable", "empty")


def test_ocr_engine_vlm_fallback(tiny_png):
    """工单四：PaddleOCR 不可用时走 VLM 转录兜底"""
    eng = ImageOCREngine(vlm_transcribe_fn=lambda im: "转录文本")
    res = eng.recognize(Image.open(tiny_png))
    assert res["ocr_text"] == "转录文本"
    assert res["engine"] == "qwen2vl"


def test_vqa_question_template_selection():
    """工单四：按类型关键词选择预置问题模板"""
    vqa = ImageVQAEngine(MockVLM())
    # 工单四：增长图 → 图表模板（含指定四问）
    qs = vqa.detect_question_set({"image_type": "", "path": "p072_img"},
                                 caption="中国IC市场应用结构与增长柱状图")
    assert qs == CHART_QUESTIONS
    assert any("增长率最快" in q for q in qs)
    assert any("负增长" in q for q in qs)
    # 工单四：组织结构图 → 结构模板
    qs2 = vqa.detect_question_set({"image_type": "", "path": ""},
                                  caption="公司组织结构图")
    assert qs2 == ORG_QUESTIONS


def test_vqa_run_returns_qa_list(tiny_png):
    """工单四：VQA 逐问执行返回 [{q,a}]"""
    vqa = ImageVQAEngine(MockVLM())
    qa = vqa.run(Image.open(tiny_png), {"image_id": "img_001"},
                 caption="增长柱状图")
    assert len(qa) == len(CHART_QUESTIONS)
    assert all("q" in x and "a" in x for x in qa)
    assert qa[2]["a"] == "汽车，14.0%"          # 工单四：增长最快模板命中 mock
    assert qa[3]["a"] == "IC卡，-2.0%"          # 工单四：负增长模板命中 mock


def test_captioner_pipeline_merge(manifest_json, tmp_path):
    """工单四：caption+VQA+OCR 全流水线合并输出 parsed JSON"""
    out = tmp_path / "parsed" / "测试册_images_parsed.json"
    cap = ImageCaptioner(vlm_engine=MockVLM(), enable_vqa=True, enable_ocr=True)
    result = cap.parse_images(str(manifest_json), str(out))

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["work_order"] == WORK_ORDER
    assert data["stats"]["total"] == 1 and data["stats"]["ok"] == 1
    img = data["images"][0]
    # 工单四：三字段齐备
    assert img["caption"].startswith("组织结构图")
    assert isinstance(img["vqa_qa"], list) and img["vqa_qa"]
    assert img["ocr_text"] == "总经理 销售部 大客户销售部 销售处"
    assert img["parse_status"] == "ok"


def test_captioner_skip_flags(manifest_json, tmp_path):
    """工单四：--skip-vqa/--skip-ocr 开关生效"""
    out = tmp_path / "parsed2.json"
    cap = ImageCaptioner(vlm_engine=MockVLM(), enable_vqa=False, enable_ocr=False)
    result = cap.parse_images(str(manifest_json), str(out))
    img = result["images"][0]
    assert img["vqa_qa"] == [] and img["ocr_text"] == ""


# ================= 工单四 Step 1（新框架）：image_parser 统一入口 ================
from src.image_parser.image_parser import CLIPEngine, ImageParser  # 工单四：新增入口


def test_clip_engine_missing_model_graceful(tiny_png, tmp_path):
    """工单四：Chinese-CLIP 模型缺失时优雅降级（不抛异常、返回 None）"""
    clip = CLIPEngine(model_dir=str(tmp_path / "no_such_clip"))
    assert clip.embed_image(Image.open(tiny_png)) is None
    assert clip.available is False


def test_image_parser_pipeline_with_embedding(manifest_json, tmp_path):
    """工单四：统一解析入口——caption/VQA/OCR 复用 + embedding 字段（模型缺失为 None）"""
    out = tmp_path / "parsed_v2" / "测试册_images_parsed.json"
    ip = ImageParser(vlm_engine=MockVLM(),
                     clip_engine=CLIPEngine(model_dir=str(tmp_path / "no_clip")))
    result = ip.parse_images(str(manifest_json), str(out))

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["work_order"] == WORK_ORDER
    assert data["stats"]["total"] == 1 and data["stats"]["ok"] == 1
    img = data["images"][0]
    assert img["caption"].startswith("组织结构图")     # 工单四：描述复用 captioner
    assert img["ocr_text"] == "总经理 销售部 大客户销售部 销售处"
    assert img["vqa_qa"] and "q" in img["vqa_qa"][0]
    assert img["embedding"] is None                    # 工单四：CLIP 缺失→None
    assert data["clip_enabled"] is False


@pytest.mark.skipif(not MODEL_DIR.exists() or os.environ.get("RUN_VLM_TESTS") != "1",
                    reason="Qwen2-VL 模型未下载或未设置 RUN_VLM_TESTS=1，跳过集成")
def test_real_qwen2vl_caption(tiny_png):
    """工单四：真实 Qwen2-VL 冒烟（显式开启才运行）"""
    eng = Qwen2VLEngine()
    out = eng.caption(Image.open(tiny_png))
    assert isinstance(out, str) and len(out) > 4
