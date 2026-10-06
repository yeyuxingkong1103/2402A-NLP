# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化任务
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
# 模块：dialog —— 多轮对话（会话状态 + 指代消解 + 省略式追问补全）
# 说明：工单 05 的核心实现。对每一轮用户提问先做指代消解/追问改写，得到「独立完整问题」，
#       再交给 engine（Query 理解 → 混合检索 → 生成）作答；同时维护会话状态，
#       使「他」「这个公司」「那 XX 公司呢」等省略表达能被正确还原。
#
# 支持的四类表达：
#   1) 完整问题（带公司全名）        -> 直接检索，并记录当前实体
#   2) 代词指代（他/她/它/这个公司…）-> 代词替换为「当前实体」
#   3) 省略式追问（「那 X 呢？」）    -> 抽出新实体 X，继承上一轮意图，拼成完整问题
#   4) 无实体无代词（「它的注册资本呢？」）-> 沿用「当前实体」
import os
import re
import sys
import time
import uuid
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine                     # noqa: E402
from config import DOC_ALIASES    # noqa: E402

# ---------- 实体与指代词典 ----------
# 文档 id -> 公司全名（问题改写时使用全名，便于 engine.detect_doc 精确路由）
ENTITY_OF_DOC = {
    "招股说明书1": "武汉兴图新科电子股份有限公司",
    "招股说明书2": "武汉力源信息技术股份有限公司",
}
# 公司全名正则（与 engine._COMPANY_RE 同口径）
CO_RE = re.compile(r"[\u4e00-\u9fff]{2,20}(?:股份)?有限公司")
# 代词/指代表达（长优先，避免「这个公司」被「该」抢先匹配）
PRON_RE = re.compile(r"这个公司|这家公司|那个公司|那家公司|该公司|本公司|贵公司|这公司|他|她|它|该(?:公司|企业)?")
# 省略式追问：整句只剩「(那)? <实体/名称> (呢/怎么样)？」
FOLLOW_RE = re.compile(
    r"^\s*(?:那|那么)?\s*([\u4e00-\u9fffA-Za-z0-9]{2,30}(?:股份)?有限公司|[\u4e00-\u9fff]{2,10})"
    r"\s*(?:呢|呢[？?]|怎么样|如何)?\s*[？?]?\s*$")


# ---------- 会话状态 ----------
class SessionState:
    """一个会话的多轮上下文。"""

    def __init__(self, sid):
        self.id = sid
        self.current_entity = None      # 当前实体（公司全名）
        self.current_doc = None         # 当前实体对应的文档 id
        self.last_intent = None         # 上一轮意图（数值/事实/列举…）
        self.last_question = None       # 上一轮原始问题
        self.last_rewrite = None        # 上一轮改写后的完整问题（作为追问的模板）
        self.turns = []                 # 轮次记录
        self.created_at = time.time()
        self.updated_at = self.created_at

    def as_dict(self):
        return {"session_id": self.id, "current_entity": self.current_entity,
                "current_doc": self.current_doc, "last_intent": self.last_intent,
                "last_question": self.last_question, "turns": len(self.turns),
                "created_at": self.created_at, "updated_at": self.updated_at}


_SESSIONS = {}
_LOCK = threading.RLock()


def new_session():
    sid = uuid.uuid4().hex[:12]
    with _LOCK:
        _SESSIONS[sid] = SessionState(sid)
    return sid


def get_state(sid):
    with _LOCK:
        if sid not in _SESSIONS:
            _SESSIONS[sid] = SessionState(sid or uuid.uuid4().hex[:12])
        return _SESSIONS[sid]


def reset_session(sid):
    with _LOCK:
        _SESSIONS.pop(sid, None)
    return True


def list_sessions():
    with _LOCK:
        return [s.as_dict() for s in _SESSIONS.values()]


# ---------- 实体识别 ----------
def detect_entity(text):
    """返回问题中出现的公司实体（全名）；无则 None。
    优先匹配完整公司名，其次按 config.DOC_ALIASES 的别名/简称映射到公司全名。"""
    if not text:
        return None
    m = CO_RE.search(text)
    if m:
        # 去掉可能被正则吃掉的前导指示词（“那/这/该/其”，如“那武汉力源…呢”），避免把“那”拼进公司名
        return re.sub(r"^(?:那|那么|这|该|其)", "", m.group(0))
    low = text.lower()
    for doc, aliases in DOC_ALIASES.items():
        for a in aliases:
            if a and a.lower() in low:
                return ENTITY_OF_DOC.get(doc, a)
    return None


def doc_of_entity(entity):
    for doc, name in ENTITY_OF_DOC.items():
        if entity and (entity == name or entity in name or name in entity):
            return doc
    return None


def _is_followup(q):
    """省略式追问判定：整句只剩「(那)? 实体 呢？」形式。"""
    s = (q or "").strip()
    if not s or len(s) > 40:
        return False
    if not detect_entity(s):
        return False
    if not FOLLOW_RE.match(s):
        return False
    # 去掉实体/语气词后应基本为空 → 说明这轮没有新意图，只是换实体
    body = s.replace(detect_entity(s) or "", "")
    for kw in ("那", "那么", "呢", "怎么样", "如何", "的", "？", "?"):
        body = body.replace(kw, "")
    return len(body.strip()) <= 2


# ---------- 指代消解 / 追问改写 ----------
def resolve(query, state):
    """把本轮问题改写为「独立完整问题」并更新会话状态。
    返回 (rewritten, info)。info 描述本轮做了哪类消解，便于前端展示/演示。"""
    q = (query or "").strip()
    info = {"type": "direct", "note": "完整问题，直接检索", "entity": None, "inherited_intent": None}
    ent = detect_entity(q)
    pron = PRON_RE.search(q)

    if _is_followup(q):
        # 省略式追问：换实体、继承上一轮「模板」
        new_ent = ent or state.current_entity
        base = state.last_rewrite or state.last_question or ""
        old_ent = state.current_entity
        if base and old_ent and new_ent:
            rewritten = base.replace(old_ent, new_ent)
        elif base and new_ent:
            rewritten = new_ent + "的" + base
        else:
            rewritten = new_ent or q
        info.update({"type": "ellipsis", "note": "省略式追问：继承上一轮意图，仅替换实体",
                     "entity": new_ent, "inherited_intent": state.last_intent})
        if new_ent:
            state.current_entity = new_ent
            state.current_doc = doc_of_entity(new_ent)

    elif pron and not ent and state.current_entity:
        # 代词指代：他/她/它/这个公司 → 当前实体
        rewritten = PRON_RE.sub(state.current_entity, q, count=1)
        info.update({"type": "pronoun", "note": "代词消解：%s → %s" % (pron.group(0), state.current_entity),
                     "entity": state.current_entity})

    elif ent:
        # 显式实体：正常问题，记录实体（可能切换公司）
        rewritten = q
        info.update({"type": "direct", "note": "显式实体：%s" % ent, "entity": ent})
        state.current_entity = ent
        state.current_doc = doc_of_entity(ent)

    else:
        # 无实体无代词：沿用当前实体
        rewritten = q
        if state.current_entity and state.current_entity not in q:
            rewritten = state.current_entity + q  # 补实体，保证 doc 路由不丢
        info.update({"type": "carry", "note": "沿用当前实体：%s" % (state.current_entity or "无")})

    # 记录本轮意图（用改写后完整问题计算，保证追问继承的是"真实意图"）
    qu = engine.query_understanding(rewritten)
    state.last_intent = qu.get("intent")
    state.last_question = q
    state.last_rewrite = rewritten
    state.updated_at = time.time()
    info["intent"] = state.last_intent
    info["doc"] = state.current_doc
    return rewritten, info


def chat(session_id, query, doc=None, top_k=None):
    """一轮对话：指代消解 → engine 作答 → 记录轮次。返回可直接 JSON 序列化的 dict。"""
    sid = session_id or new_session()
    state = get_state(sid)
    t0 = time.time()
    rewritten, info = resolve(query, state)
    r = engine.answer(rewritten, doc=doc, top_k=top_k) if top_k else engine.answer(rewritten, doc=doc)
    cost_ms = int((time.time() - t0) * 1000)
    turn = {"q": query, "rewritten": rewritten, "answer": r["answer"],
            "refused": r.get("refused"), "confidence": r.get("confidence"),
            "resolution": info, "cost_ms": cost_ms}
    state.turns.append(turn)
    return {"session_id": sid, "query": query, "rewritten_query": rewritten,
            "resolution": info, "intent": info.get("intent"),
            "answer": r["answer"], "refused": r.get("refused"),
            "confidence": r.get("confidence"), "doc": r.get("doc"),
            "cost_ms": cost_ms,
            "hits": [{"doc": h.get("doc"), "page": h["page"], "page_end": h.get("page_end"),
                      "section": h.get("section"), "type": h.get("type"),
                      "sim": h.get("dense_sim"), "text": h["text"]} for h in r.get("hits", [])]}


def run_dialog(questions, session_id=None):
    """顺序跑一段多轮对话（用于演示/评估）。返回每轮结果。"""
    sid = session_id or new_session()
    return [chat(sid, q) for q in questions]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    DEMO = [
        "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        "他参与的哪个工程荣获了国家科技进步一等奖？",
        "这个公司的法定代表人是谁？",
        "那武汉力源信息技术股份有限公司呢？",
        "武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
    ]
    for i, r in enumerate(run_dialog(DEMO), 1):
        print("=" * 70)
        print("第 %d 轮  Q: %s" % (i, r["query"]))
        print("      改写: %s" % r["rewritten_query"])
        print("      消解: %s" % r["resolution"]["note"])
        print("      A(%dms, conf=%s): %s" % (r["cost_ms"], r["confidence"], r["answer"][:300]))
