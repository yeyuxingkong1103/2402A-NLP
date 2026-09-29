# -*- coding: utf-8 -*-
"""核心单元测试（一）：工程约束守护 + BM25 + 入库管线 + Prompt/角色。

不连任何外部服务（Milvus/Redis/LLM），全部纯逻辑可离线跑。
"""
import sys  # 标准库：改模块搜索路径
from pathlib import Path  # 拼项目根路径

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 本文件在 tests/ 下，向上一级即项目根
sys.path.insert(0, str(PROJECT_ROOT))  # 插到最前：pytest 从任意目录运行都能 import src


# ========== 工程约束守护（老师要求：单文件 ≤150 行、src ≤24 个文件）==========

def test_no_file_exceeds_150_lines():  # 守护测试：单文件行数上限，倒逼代码精简
    """每个 py 文件（含 src/scripts/tests）不超 150 行。"""
    violations = []  # 收集违规文件
    for path in PROJECT_ROOT.rglob("*.py"):  # 递归扫全项目 py 文件
        rel = path.relative_to(PROJECT_ROOT).as_posix()  # 转相对路径（posix 风格跨平台一致）
        if "__pycache__" in rel:  # 跳过缓存目录
            continue  # 缓存文件不算源码
        lines = len(path.read_text(encoding="utf-8").splitlines())  # 按行切分统计行数
        if lines > 150:  # 超阈值即违规
            violations.append(f"{rel}: {lines}行")  # 记下文件名与实际行数
    assert not violations, f"以下文件超150行：{'; '.join(violations)}"  # 断言为空；失败信息列出所有违规文件


def test_files_total_within_reasonable():  # 守护测试：src 文件数上限，防无限膨胀
    """核心 src/ 不超 24 个文件（保持精简）。"""
    src_files = list((PROJECT_ROOT / "src").rglob("*.py"))  # 只统计 src/ 下的 py
    assert len(src_files) <= 24, f"src/ 下有 {len(src_files)} 个文件，要求 ≤ 24"  # 超过即报错并提示实际数量


# ========== BM25 关键词检索 ==========

def test_bm25_tokenization():  # 验证 BM25 中文分词与排序打分
    """BM25 中文分词与打分正常。

    语料必须够大：BM25Okapi 的 IDF 在只有 2 篇文档时会退化为负值
    （N=2 时 log((N-df+0.5)/(df+0.5)) 对高频词为负，epsilon 兜底也是负的）。
    """
    from src.bm25_index import BM25Index  # 函数内导入：被测模块加载失败只影响本测试

    bm25 = BM25Index()  # 空索引
    bm25.build([  # 造 6 篇小语料建索引（docstring 已说明为何语料要够大）
        {"text": "高血压患者应低盐饮食", "source": "指南A", "page": 1},  # 文档0：含"高血压"
        {"text": "高血压常用药物有CCB、ACEI", "source": "指南A", "page": 2},  # 文档1：唯一同时含两关键词
        {"text": "规律运动有助于降低血压", "source": "指南A", "page": 3},  # 文档2：只含部分词
        {"text": "戒烟限酒对心血管有益", "source": "指南A", "page": 4},  # 文档3：干扰项，两关键词都不含
        {"text": "保持情绪平稳有助于血压控制", "source": "指南A", "page": 5},  # 文档4：干扰项
        {"text": "定期监测血压并做好记录", "source": "指南A", "page": 6},  # 文档5：干扰项，凑足语料规模
    ])  # 建索引完成
    results = bm25.search("高血压药物", top_k=2)  # 双关键词查询取前 2
    assert len(results) == 2  # 数量正确
    assert results[0][0] == 1  # 唯一同时含"高血压"与"药物"的一篇应排第一
    assert results[0][1] > 0  # 得分必须为正（IDF 退化会给负分，见 docstring）


# ========== 入库管线 ==========

def test_split_pages():  # 验证按页标记切分
    """按“[第N页]”标记切页正确。"""
    from src.ingestion import split_pages  # 被测函数

    pages = split_pages("前言[第1页]第一页正文[第3页]第三页正文")  # 含跳页（1→3）与前言噪声的样例
    assert pages == [(1, "第一页正文"), (3, "第三页正文")]  # 前言被丢弃、页码与正文一一对应


def test_chunk_pages_produces_output():  # 验证切块结构完整
    """切块必须带来源与页码（引用展示要用）。"""
    from src.ingestion import chunk_pages  # 被测函数

    pages = [(1, "高血压是一种常见慢性病，需要长期规范管理。" * 6)]  # 单页长文本：重复 6 倍保证切出多块
    chunks = chunk_pages(pages, source="测试指南")  # 切块
    assert len(chunks) >= 2  # 长文本必须切出至少 2 块
    assert all(set(c) == {"text", "source", "page"} for c in chunks)  # 每块恰好三个字段：正文/来源/页码
    assert all(c["source"] == "测试指南" and c["page"] == 1 for c in chunks)  # 来源与页码必须随块传递（引用溯源要用）


# ========== Prompt 与角色 ==========

def test_prompts_format_history():  # 验证短期记忆格式化
    """短期记忆格式化逻辑正确。"""
    from src import prompts  # 被测模块

    history = [  # 造一轮问答历史
        {"role": "user", "content": "我头疼"},  # 用户发言
        {"role": "assistant", "content": "请问多久了？"},  # 助手回复
    ]  # 历史结束
    result = prompts.format_history(history)  # 格式化成提示词里的历史段
    assert "用户：我头疼" in result  # role 要转成中文标签
    assert "助手：请问多久了" in result  # assistant 显示为"助手"


def test_intent_mode_knowledge_forbids_followup():  # 验证意图模式话术
    """知识科普类必须直接作答、禁止反问。"""
    from src import prompts  # 被测模块
    from src.intent import KNOWLEDGE, PERSONAL  # 两类意图常量

    assert "禁止反问" in prompts.intent_mode(KNOWLEDGE)  # 知识类直接答，不许追问
    assert "先结论" in prompts.intent_mode(PERSONAL)  # 个人病情类先给结论再追问


def test_role_prompt_contains_guardrails():  # 验证角色提示词带安全护栏
    """角色 Prompt 必须包含安全护栏关键词。"""
    from src.role import RoleManager  # 被测类

    prompt = RoleManager().build_system_prompt("doctor")  # 构建医生人设系统提示词
    assert "不能替代医生面诊" in prompt  # 必须有免责护栏
    assert "⚠️" in prompt  # 必须有警示符号（急症/就医提示）


def test_psychologist_prompt_bans_tcm_terms():  # 验证心理角色禁用中医术语
    """心理医生不得出现中医辨证术语。"""
    from src.role import RoleManager  # 被测类

    prompt = RoleManager().build_system_prompt("psychologist")  # 构建心理医生人设
    assert "禁用术语" in prompt and "肝阳上亢" in prompt  # 禁用清单要明文写进提示词，防止角色串味
