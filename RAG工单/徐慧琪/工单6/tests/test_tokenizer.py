# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务

测试：中英混合分词与数字口径归一（src/tokenizer.py）
不加载模型、不联网（jieba 词典为本地自带）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import tokenizer  # noqa: E402


def test_protected_terms_kept_whole():
    """领域专名必须整体成词（jieba 会切错，见 tokenizer 注释）。"""
    toks = tokenizer.tokenize("公司注册资本为5,520.00万元")
    assert "注册资本" in toks


def test_english_and_numbers_lowercased():
    toks = tokenizer.tokenize("BGE-M3 模型 test 25.04%")
    assert "bge-m3" in toks
    assert "test" in toks
    assert "25.04%" in toks


def test_number_forms_decimal_zero_trimmed():
    """数字口径归一：5,520.00 必须能等同 5,520 检索（招股书数字是主干）。"""
    toks = tokenizer.tokenize("5,520.00万元")
    assert "5,520.00" in toks
    assert "5,520" in toks


def test_number_forms_thousands_stripped():
    toks = tokenizer.tokenize("1,670万股")
    assert "1,670" in toks
    assert "1670" in toks


def test_number_forms_keep_meaningful_decimals():
    """非零小数不该被裁剪（25.04% 与 25% 是不同事实）。"""
    toks = tokenizer.tokenize("25.04%")
    assert "25.04%" in toks
    assert "25%" not in toks


def test_query_stopwords_dropped_only_in_query_side():
    assert "的" in tokenizer.tokenize("公司的注册资本")
    assert "的" not in tokenizer.tokenize_query("公司的注册资本")
    # 保护词不受停用词影响
    assert "注册资本" in tokenizer.tokenize_query("公司的注册资本")


def test_tokenize_empty():
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize_query("") == []
