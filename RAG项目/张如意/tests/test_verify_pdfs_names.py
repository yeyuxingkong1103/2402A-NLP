# -*- coding: utf-8 -*-
"""verify_pdfs 的文件名解析单测：确保非 GB 命名（如 CCPIA 公告）不会让它崩。"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 按文件路径动态加载被测模块（scripts/verify_pdfs.py），不依赖包安装方式
spec = importlib.util.spec_from_file_location("vp", ROOT / "scripts/verify_pdfs.py")
vp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp)


def test_国标文件名拆出标准号与名称():
    """GB_Z / GB_T 命名的文件（下划线代替斜杠）要能还原成标准号 + 标准名称两部分。"""
    assert vp.split_name("GB_Z 26573-2011 菠菜生产技术规范") == ("GB/Z 26573-2011", "菠菜生产技术规范")
    assert vp.split_name("GB_T 22000-2006 在饲料加工企业的应用指南") == \
        ("GB/T 22000-2006", "在饲料加工企业的应用指南")


def test_团标文件名可识别():
    """团体标准（T/CCPIA 前缀）同样要能拆出标准号与名称。"""
    assert vp.split_name("T_CCPIA 262-2025 小麦安全科学使用农药指南") == \
        ("T/CCPIA 262-2025", "小麦安全科学使用农药指南")


def test_识别不出的文件名不崩():
    """非标准命名的文件（如协会公告）：标准号返回空串、名称回退为文件名本身，不许抛异常。"""
    stdno, name = vp.split_name("中国农药工业协会关于发布《水稻安全科学使用农药指南》等6项团体标准的公告")
    assert stdno == "" and name.startswith("中国农药工业协会")


def test_归一化用于文本比对():
    """norm 用于把「GB/Z 26573—2011」（全角破折号、空格、斜杠）归一成可与正文比对的紧凑形式。"""
    assert vp.norm("GB/Z 26573—2011") == "GBZ265732011"
