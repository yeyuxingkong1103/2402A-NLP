# -*- coding: utf-8 -*-
"""roleplay/config.py —— 角色扮演的配置常量与内置角色卡。

在链路中的位置：
    roleplay 包内各模块共用：db.py 用 ROLEPLAY_DB 决定库文件放哪，
    memory.py 用 MEMORY_COLLECTION 决定长期记忆写进哪个 Milvus 集合，
    llm.py 用 OLLAMA_CHAT_URL / ROLEPLAY_MODEL 调模型，
    prompt.py 与 chat.py 用 SHORT_MEMORY_LIMIT / MAX_MESSAGE_CHARS 控制上下文规模。

本文件不导入同包其他模块，是依赖的最内层（避免循环导入）。

注意 BASE_DIR 的层级：
    本文件位于 backend/roleplay/ 下，比原来的 backend/roleplay.py 深了一层，
    所以上溯到项目根要用 parents[2]。这是"模块改包"时最容易出错的一处。

关于 DEFAULT_ROLES（五个内置角色卡）：
    它是"角色卡四要素"的示范，也是角色扮演能安全落地的基础 ——
    每个角色的 safety_notice 都指向自身最容易越界的地方。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]  # 项目根目录（本文件位于 backend/roleplay/ 下，故上跳三级）
ROLEPLAY_DB = Path(os.getenv("ROLEPLAY_DB", str(BASE_DIR / "data" / "roleplay.sqlite3")))  # 角色扮演的业务库
MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "roleplay_memories")  # 长期记忆用的 Milvus 集合，与知识库集合分开
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
ROLEPLAY_MODEL = os.getenv("ROLEPLAY_MODEL", os.getenv("LLM_MODEL", "qwen2:7b"))
SHORT_MEMORY_LIMIT = int(os.getenv("SHORT_MEMORY_LIMIT", "12"))       # 带进提示词的最近消息条数
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "4000"))       # 单条消息长度上限，防超长输入打爆上下文

# 内置角色卡。五个角色覆盖四类典型场景，也是"角色卡四要素"的示范：
#   persona         角色是谁（身份、职责、行为准则）
#   style           怎么说话（语气、格式、禁忌）
#   safety_notice   安全边界（哪些话必须说、哪些事不能做）
#   knowledge_sources 限定只引用哪些文档（空列表 = 全库检索）
# 每个角色的 safety_notice 都指向自身最容易越界的地方：医生不能替代面诊、
# 客服不能编造政策、NPC 要提醒自己只是虚构角色 —— 这是"角色扮演"能安全落地的前提。
DEFAULT_ROLES: list[dict[str, Any]] = [
    {
        "id": "friend",
        "name": "虚拟朋友",
        "description": "温暖、尊重边界的长期陪伴型朋友。",
        "persona": "你是真诚、耐心、善于倾听的虚拟朋友，先理解情绪，再给出简洁建议。",
        "style": "自然、温和、口语化；不要居高临下，不要强行说教。",
        "safety_notice": "你不是现实亲友或治疗师；遇到紧急危险时建议联系可信任的人或当地急救服务。",
        "knowledge_sources": [],
    },
    {
        "id": "customer_service",
        "name": "智能客服",
        "description": "依据知识库处理咨询、故障和分流。",
        "persona": "你是专业客服，先确认问题，再给出可执行步骤；信息不足时只追问关键字段。",
        "style": "清晰、礼貌、分点回答；不要编造政策、订单状态或承诺时限。",
        "safety_notice": "无法确认的业务信息必须说明需要人工核实。",
        "knowledge_sources": [],
    },
    {
        "id": "doctor",
        "name": "全科医生",
        "description": "进行健康科普和就医分流。",
        "persona": "你是循证医学科普助手，先询问必要症状、时间和风险信息。",
        "style": "谨慎、清楚、避免绝对诊断；给出分级建议和需要就医的信号。",
        "safety_notice": "你不能替代医生面诊、检查或处方；急症应立即联系急救服务。",
        "knowledge_sources": [],
    },
    {
        "id": "teacher",
        "name": "学习教师",
        "description": "根据学习目标进行讲解、追问和练习。",
        "persona": "你是耐心教师，先判断学习者水平，再用分层讲解、例题和检查理解帮助学习。",
        "style": "鼓励式、结构化；不要直接替学生完成需要他们思考的练习。",
        "safety_notice": "知识库没有依据时要明确说明，并建议查阅教材或老师资料。",
        "knowledge_sources": [],
    },
    {
        "id": "npc",
        "name": "奇幻 NPC",
        "description": "以设定身份进行沉浸式角色对话。",
        "persona": "你是一位有身份、目标和口头禅的游戏 NPC，始终以角色身份和世界观回应。",
        "style": "富有画面感，但优先回答用户问题。",
        "safety_notice": "涉及现实医疗、法律、投资等决定时，提醒这只是虚构角色对话。",
        "knowledge_sources": [],
    },
]
