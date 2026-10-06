# -*- coding: utf-8 -*-
"""
问答引擎 V5 - 多轮对话整合
工单编号: 人工智能 NLP-RAG-Query 理解优化任务

核心: DialogueManager 补全问题 → 调用 V4 检索管线
"""
import os
import sys
import time
import logging
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config_v5 as config
import dialogue_manager

logger = logging.getLogger(__name__)

# 引入 V4 引擎 (图像+表格+文本)
_V4_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "工单4", "研发"))
if _V4_DIR not in sys.path:
    sys.path.insert(0, _V4_DIR)


def _call_v4_answer(question: str) -> Dict:
    """调用 V4 完整 RAG 管线"""
    try:
        import qa_engine_v4
        return qa_engine_v4.answer_question(question)
    except Exception as e:
        logger.warning(f"V4 引擎不可用: {e}")
        # 降级: 尝试 V3
        try:
            _V3_DIR = os.path.abspath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "工单3", "研发"))
            sys.path.insert(0, _V3_DIR)
            import qa_engine_v3
            return qa_engine_v3.answer_question(question)
        except Exception as e2:
            logger.error(f"所有引擎不可用: {e2}")
            return {"rag_answer": f"引擎不可用: {e}", "llm_answer": "",
                    "response_time": 0, "within_time_limit": False}


_dm = dialogue_manager.DialogueManager()


def answer_question(question: str, session_id: str = "default") -> Dict:
    """
    V5 多轮问答主入口

    Args:
        question: 用户当前问题
        session_id: 会话 ID (用于区分不同用户)

    Returns:
        {
            "question": str,        # 原始问题
            "enriched": str,        # 补全后的问题
            "dialogue_info": Dict,  # 会话状态
            "rag_answer": str,
            "llm_answer": str,
            "response_time": float,
            "within_time_limit": bool,
        }
    """
    start = time.time()

    # 1. Query 理解 (指代消解 + 省略补全)
    understood = _dm.understand_query(question, session_id)
    enriched = understood["enriched"]

    # 2. 调用底层检索引擎 (用补全后的问题)
    v4_result = _call_v4_answer(enriched)

    # 3. 更新会话
    ctx = _dm.get_session(session_id)
    ctx.record(
        question=question,
        enriched=enriched,
        answer=v4_result.get("rag_answer", ""),
        topic=understood.get("topic", ""),
    )

    elapsed = time.time() - start

    return {
        "question": question,
        "enriched": enriched,
        "dialogue_info": {
            "session_id": session_id,
            "turn": understood["turn_count"],
            "current_company": understood["company"],
            "company_switched": understood["company_switched"],
            "topic": understood.get("topic", ""),
            "is_multiturn": ctx.turn_count > 1,
        },
        "rag_answer": v4_result.get("rag_answer", ""),
        "llm_answer": v4_result.get("llm_answer", ""),
        "response_time": round(elapsed, 3),
        "within_time_limit": elapsed <= 3.0,
        "answer_source": v4_result.get("answer_source", ""),
    }


def new_session(session_id: str = "default") -> Dict:
    """重置会话"""
    _dm._sessions[session_id] = dialogue_manager.SessionContext(session_id)
    return {"session_id": session_id, "reset": True}


def get_session_context(session_id: str = "default") -> Dict:
    """获取当前会话状态"""
    return _dm.get_session(session_id).to_dict()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sid = "test_dialogue"
    dialogues = [
        "报告期内,武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少?",
        "他参与的哪个工程荣获了国家科技进步一等奖?",
        "这个公司的法定代表人是谁?",
        "那武汉力源信息技术股份有限公司呢?",
        "武汉力源信息技术股份有限公司组织结构图中,哪个销售部的销售处最多?",
    ]

    for q in dialogues:
        print(f"\n{'='*60}")
        r = answer_question(q, sid)
        di = r["dialogue_info"]
        print(f"[{di['turn']}] {q[:50]}")
        print(f"  补全: {r['enriched'][:60]}")
        print(f"  公司: {di['current_company']} | 话题: {di['topic']} | 切换: {di['company_switched']}")
        print(f"  RAG: {r['rag_answer'][:100]}")
