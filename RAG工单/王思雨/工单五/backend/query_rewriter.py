# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：多轮对话 Query 理解（工单五核心）。每轮先用大模型把省略/指代的问句改写成可独立检索的问句，
#           同时做实体追踪：用户切换公司后，"他/这个公司"必须指向新公司。
#           改写失败时用规则兜底，保证检索仍能进行，并如实上报 degraded。
import json                                                                     # 实体表持久化
import re                                                                       # 实体识别与规则兜底
from typing import Dict, List, Tuple                                            # 类型标注

from . import config as C                                                       # 复用集中配置
from . import deepseek_llm, redis_cache                                         # 大模型与会话存储

# 公司全称的形态：以"股份有限公司/有限公司/集团"结尾的中文名
_COMPANY = re.compile(r"[一-龥A-Za-z（）()]{2,30}?(?:股份有限公司|有限责任公司|有限公司|集团公司|集团)")
# 指代词的常见写法（工单明确点名"他""这个公司"）
_PRONOUN = re.compile(r"他|她|它|该?公司|这家公司|该公司|这个公司|此公司")
# "那X呢 / X呢" 这类切换实体的问法
_SWITCH = re.compile(r"^那?么?(?P<rest>.{0,40}?)(?:呢|呢？|\?|？)?$")

_ENTITIES: List[str] = []                                                       # 已知实体全称（长优先匹配）
_PROMPT = (                                                                     # 改写用提示词
    "你是多轮对话的查询改写器。请把【最新提问】改写成一句可独立检索的完整问句，规则：\n"
    "1) 把“他/她/它/公司/这个公司/该公司”等指代，替换成【上文】中正在讨论的公司全称；\n"
    "2) 若用户用“那X呢/X呢”切换了公司，则以新公司为准，并把上一轮的问题主题(如“法定代表人是谁”“军用领域收入”)\n"
    "   套用到新公司上；\n"
    "3) 不要改变用户的原意，也不要补充用户没问的内容；\n"
    "4) 只输出改写后的问句本身，不要解释、不要加引号、不要输出任何其它文字。"
)


def load_entities(path=None) -> List[str]:                                      # 读取实体表
    """实体表由入库脚本生成（data/entities.json）；缺失时退回内置的两家公司全称与简称。"""
    global _ENTITIES                                                            # 声明全局
    path = path or (C.ROOT / "data" / "entities.json")                          # 默认路径
    names: List[str] = []                                                       # 结果
    try:                                                                        # 读盘可能失败
        if path.exists():                                                       # 文件存在
            names = json.loads(path.read_text(encoding="utf-8"))                # 反序列化
    except Exception as exc:                                                    # 文件损坏
        print(f"[Query改写] 实体表读取失败：{type(exc).__name__}: {exc}")        # 记录
    if not names:                                                               # 没有任何实体
        names = ["武汉兴图新科电子股份有限公司", "武汉力源信息技术股份有限公司",      # 两篇招股书的公司全称
                 "兴图新科", "力源信息"]                                          # 与常用简称
    _ENTITIES = sorted(set(names), key=len, reverse=True)                       # 长名优先，避免短名抢先命中
    return _ENTITIES                                                            # 返回


def entities() -> List[str]:                                                    # 取实体表
    """未加载时自动加载，避免调用方关心初始化顺序。"""
    return _ENTITIES or load_entities()                                         # 返回（必要时先加载）


def detect_entity(text: str) -> str:                                            # 实体识别
    """从文本中识别公司实体：先试已知实体表（长优先），再用全称正则兜底，最后识别简称。"""
    t = text or ""                                                              # 防空
    for name in entities():                                                     # 已知实体逐个匹配
        if name in t:                                                           # 命中
            return name                                                         # 返回全称/简称
    m = _COMPANY.search(t)                                                      # 正则兜底：形如"XX股份有限公司"
    return m.group(0) if m else ""                                              # 命中返回，否则空串


def _rule_rewrite(question: str, history: List[Dict], entity: str,              # 规则兜底改写
                  switched: str) -> str:                                        # switched 为本轮新实体
    """大模型不可用时的降级路径：用最近一轮问句做模板，把实体替换成当前实体。"""
    target = switched or entity                                                 # 本轮指向的公司
    last_q = ""                                                                 # 上一轮用户问句
    for h in reversed(history):                                                 # 从最新往回找
        if h.get("role") == "user":                                             # 只取用户问句
            last_q = str(h.get("content", ""))                                  # 记录
            break                                                               # 找到即停
    if switched and last_q:                                                     # 用户切换了公司 → 套用上轮主题
        theme = re.sub(r"^(那|那么)?\s*" + re.escape(switched) + r"\s*(呢|呢？|\?|？)?$", "", question).strip()  # 去掉"那X呢"
        theme = theme or _theme_of(last_q)                                       # 取不到就用上轮主题
        return f"{switched}{'的' + theme if theme else ''}？"                     # 拼成独立问句
    if target:                                                                  # 有明确实体
        out = _PRONOUN.sub(target, question)                                    # 把指代词替换成实体
        if target not in out:                                                   # 原句没有指代词也没实体
            out = f"{target} {question}"                                        # 前置实体，帮助检索命中
        return out.strip()                                                      # 返回
    return question.strip()                                                     # 无实体可用时原样返回


def _theme_of(q: str) -> str:                                                   # 抽取问句主题
    """把"这个公司的法定代表人是谁？"压成"法定代表人是谁"，便于套用到新公司。"""
    s = _PRONOUN.sub("", q or "")                                               # 去掉指代词
    s = re.sub(r"^(的|其)+", "", s.strip())                                      # 去掉残留的"的/其"
    return s.strip("？?。. ")                                                     # 去掉句末标点


def rewrite(question: str, session_id: str) -> Tuple[str, str, List[str]]:      # 改写主入口
    """返回 (改写后问句, 本轮实体全称, 降级原因列表)。

    流程：识别实体 → 比对会话中记录的实体（切换则更新）→ 有历史或多轮指代时调大模型改写 → 失败走规则兜底。
    单轮、且问句自带实体且无指代词时直接返回原句，省掉一次大模型调用（对 3 秒响应目标很关键）。
    """
    degraded: List[str] = []                                                    # 降级记录
    if not C.REWRITE_ENABLED:                                                   # 开关关闭
        return question, detect_entity(question), ["rewrite"]                   # 直接用原句
    history = redis_cache.get_history(session_id)                               # 取会话历史
    prev_entity = redis_cache.get_entity(session_id)                            # 取上一轮实体
    found = detect_entity(question)                                             # 本轮显式提到的实体
    switched = found if (found and prev_entity and found != prev_entity) else ""  # 用户这轮切换了公司
    entity = found or prev_entity                                               # 本轮实际指向的公司
    if entity and entity != prev_entity:                                        # 实体发生变化
        redis_cache.set_entity(session_id, entity)                              # 更新会话实体，供后续"他"指向新公司
    has_pronoun = bool(_PRONOUN.search(question))                               # 是否含指代词
    if not history and not has_pronoun:                                         # 首轮且无指代 → 无需改写
        return question.strip(), entity, degraded                               # 直接返回，省一次调用
    if not deepseek_llm.llm_ready():                                            # 大模型未配置
        degraded.append("rewrite")                                              # 记降级
        return _rule_rewrite(question, history, entity, switched), entity, degraded  # 规则兜底
    ctx = _history_text(history)                                                # 组装上文
    msgs = [{"role": "system", "content": _PROMPT},                             # 系统提示：只输出改写后问句
            {"role": "user", "content": f"【上文】\n{ctx}\n\n【最新提问】\n{question}"}]  # 上文与最新提问
    try:                                                                        # 改写调用可能超时或失败
        out = deepseek_llm.chat(msgs, timeout=C.REWRITE_TIMEOUT,                # 改写要快
                                max_tokens=120, temperature=0.0)                # 短输出、零温度，保证可复现
        out = _sanitize(out)                                                    # 去掉模型可能多输出的引号/前缀
        if not out:                                                             # 改写结果为空
            raise RuntimeError("改写结果为空")                                    # 触发兜底
        if entity and entity not in out and has_pronoun:                        # 模型漏掉了实体
            out = f"{entity} {out}"                                             # 补上前缀，保证检索能定位公司
        return out, entity, degraded                                            # 返回改写结果
    except Exception as exc:                                                    # 调用失败
        print(f"[Query改写] 大模型改写失败：{type(exc).__name__}: {exc}")          # 明确报因
        degraded.append("rewrite")                                              # 记降级
        return _rule_rewrite(question, history, entity, switched), entity, degraded  # 规则兜底


def _history_text(history: List[Dict]) -> str:                                  # 组装上文
    """只取最近数轮，并截断单条长度；上文越长改写越慢，对 3 秒目标不利。"""
    parts = []                                                                  # 结果
    for h in history[-C.REWRITE_MAX_HISTORY * 2:]:                              # 最近若干轮
        role = "用户" if h.get("role") == "user" else "助手"                       # 角色中文名
        parts.append(f"{role}：{str(h.get('content', ''))[:C.SESSION_MAX_CHARS]}")  # 截断后拼接
    return "\n".join(parts) or "（无上文）"                                        # 空历史显式写出


def _sanitize(text: str) -> str:                                                # 清洗改写输出
    """模型偶尔会带引号、前缀或换行；这里只保留第一行的问句本体。"""
    s = (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""  # 取首个非空行
    s = re.sub(r"^(改写后|改写结果|问题)[:：]\s*", "", s)                          # 去掉"改写后："之类前缀
    return s.strip().strip("“”\"'").strip()                                       # 去掉成对引号与首尾空白
