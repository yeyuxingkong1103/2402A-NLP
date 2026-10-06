# -*- coding: utf-8 -*-
"""离线级：``retrieval_utils.is_evidence_hit`` 的**形状鲁棒性**（A5 发现，captain 已派 t14）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

背景（captain 独立复现 + 本仓复现）：``is_evidence_hit`` 用 ``getattr(chunk, "content", None)``
取正文 —— 只支持**带 ``.content`` 属性的对象**；传 **dict** 时取到 ``None`` →
**静默返回 False**，把「形状不匹配」伪装成「真实未命中」。这类静默失败会污染命中率统计，
必须显式修复（要么支持 dict，要么抛错/记日志，绝不能返回 False 了事）。

【性质】产品缺陷（captain 已派 t14 修复）；修复后本文件应整体转绿。
测试侧已用 ``assertions._ChunkLike`` 包装规避，但**产品 API 自身**必须修。
"""

from __future__ import annotations

import types

import pytest

from common import paths

pytestmark = [pytest.mark.offline, pytest.mark.linkage]

#: 一段真实证据文本（取自 PDF1 物理 129 页，用于构造命中场景）
EVIDENCE = "公司来自军用领域的收入分别为6,464.51 万元、14,414.16 万元"
CHUNK_CONTENT = "报告期内，" + EVIDENCE + "，占主营业务收入比重分别为82.10%、97.31%、94.84%和94.34%。"
MISS_CONTENT = "本公司主要从事视频指挥控制类产品的研发、生产与销售。"


@pytest.fixture(scope="module")
def retrieval_utils() -> object:
    """产品检索工具模块（命中判定唯一入口）。"""
    paths.ensure_dev_on_path()
    from app.core import retrieval_utils as module  # noqa: PLC0415

    return module


def test_object_shape_hits(retrieval_utils: object) -> None:
    """基准：带 ``.content`` 的对象形状必须命中（这一条一直是对的）。"""
    obj = types.SimpleNamespace(content=CHUNK_CONTENT, chunk_id="t8-obj-1")
    assert retrieval_utils.is_evidence_hit([obj], EVIDENCE) is True


def test_dict_shape_must_not_silently_miss(retrieval_utils: object) -> None:
    """**dict 形状必须与对象形状给出同一判定**：不得因形状不同而静默返回 False。

    【性质：产品缺陷】修复前本用例为红（dict → False）；t14 修复后转绿。
    """
    obj = types.SimpleNamespace(content=CHUNK_CONTENT, chunk_id="t8-obj-2")
    dct = {"content": CHUNK_CONTENT, "chunk_id": "t8-dict-2"}
    obj_hit = retrieval_utils.is_evidence_hit([obj], EVIDENCE)
    dict_hit = retrieval_utils.is_evidence_hit([dct], EVIDENCE)
    assert dict_hit is True, (f"dict 形状被静默判为未命中：对象形状={obj_hit}，dict 形状={dict_hit}；"
                              f"根因=getattr(chunk,'content',None) 对 dict 取不到值却直接返回 False"
                              f"【性质：产品缺陷，t14 修复中】")


def test_dict_shape_control_case_stays_false(retrieval_utils: object) -> None:
    """对照：内容确实不含证据时，两种形状都必须判 False（防止「一律 True」式修法）。"""
    obj = types.SimpleNamespace(content=MISS_CONTENT, chunk_id="t8-obj-3")
    dct = {"content": MISS_CONTENT, "chunk_id": "t8-dict-3"}
    assert retrieval_utils.is_evidence_hit([obj], EVIDENCE) is False
    assert retrieval_utils.is_evidence_hit([dct], EVIDENCE) is False
