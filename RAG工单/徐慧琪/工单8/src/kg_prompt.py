# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：外置 Prompt 的加载与渲染

为什么外置：工单09 要在「Prompt 构建实体、关系」层面做优化并对比优化前后效果，
Prompt 若硬编码在 .py 里，优化实验就得改代码、易出错、难复现。
外置为 YAML 后，工单09 只需替换 prompts/kg_extract.yaml 即可做 A/B 对比。

同时本模块承担**防幻觉闸门**：LLM 输出的每条关系，
其 evidence 必须真实出现在原文中，否则丢弃（见 to_entities_relations）。
"""
from __future__ import annotations

import json
import os
import re
import threading

from src import config
from src.kg_schema import (ENTITY_TYPES, RELATION_TYPES, Entity, Relation,
                           validate_entity, validate_relation)

_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.RLock()


def _prompt_path(name: str) -> str:
    return os.path.join(config.PROMPTS_DIR, f"{name}.yaml")


def load_prompt(name: str = "kg_extract") -> dict:
    """读取 Prompt YAML（带 mtime 缓存，文件改动后自动重载）。"""
    path = _prompt_path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt 文件不存在：{path}")

    mtime = os.path.getmtime(path)
    with _lock:
        cached = _cache.get(name)
        if cached and cached[0] == mtime:
            return cached[1]

    try:
        import yaml
    except ImportError:                       # pragma: no cover
        raise RuntimeError("需要 PyYAML：pip install pyyaml（本机已随其他依赖安装）")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt YAML 结构异常：{path}")

    with _lock:
        _cache[name] = (mtime, data)
    return data


def _render_type_block() -> tuple[str, str]:
    ent = "\n".join(f"  - {t}：{d}" for t, d in ENTITY_TYPES.items())
    rel = "\n".join(
        f"  - {t}（{' / '.join(s['head'])} → {' / '.join(s['tail'])}）：{s['desc']}"
        for t, s in RELATION_TYPES.items())
    return ent, rel


def build_extract_messages(chunk: dict, prompt: dict | None = None) -> list[dict]:
    """构造 Ollama /api/chat 的 messages。"""
    prompt = prompt or load_prompt("kg_extract")
    ent_block, rel_block = _render_type_block()

    system = prompt["system"].format(entity_types=ent_block,
                                     relation_types=rel_block)
    heading = " / ".join(str(h) for h in (chunk.get("heading_path") or [])) or "（无）"
    user = prompt["user_template"].format(
        source=chunk.get("source", ""), page=chunk.get("page_idx", -1),
        heading=heading, text=(chunk.get("text") or "")[:1800])
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _heading_of(chunk: dict) -> str:
    return " / ".join(str(h) for h in (chunk.get("heading_path") or [])) or "（无）"


def _chunk_text(chunk: dict) -> str:
    return (chunk.get("text") or "")[:1800]


def build_entities_messages(chunk: dict, prompt: dict | None = None) -> list[dict]:
    """两段式①：只抽实体的有界调用（小输出，避免模型退化截断）。"""
    prompt = prompt or load_prompt("kg_extract")
    ent_block, _ = _render_type_block()

    system = prompt["entities_system"].format(entity_types=ent_block)
    user = prompt["entities_user_template"].format(
        source=chunk.get("source", ""), page=chunk.get("page_idx", -1),
        heading=_heading_of(chunk), text=_chunk_text(chunk))
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _render_entity_list(entities) -> str:
    """已接受实体 → 「名称（类型）」清单，供关系调用逐字复制。

    同名去重：模型在实体调用里可能重复输出同一实体（退化），
    清单里只列一次，避免误导关系调用重复成对。
    """
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for e in entities or []:
        if isinstance(e, Entity):
            name, etype = e.name, e.type
        elif isinstance(e, dict):
            name, etype = e.get("name", ""), e.get("type", "")
        else:
            name, etype = str(e), ""
        if (name, etype) in seen:
            continue
        seen.add((name, etype))
        lines.append(f"- {name}（{etype}）")
    return "\n".join(lines) or "（无）"


def build_relations_messages(chunk: dict, entities, prompt: dict | None = None
                             ) -> list[dict]:
    """两段式②：只抽关系的调用。entities 为①已接受的实体（Entity/名称均可），
    喂回去作为端点白名单 —— 关系头尾名称必须逐字从这个清单里选。"""
    prompt = prompt or load_prompt("kg_extract")
    _, rel_block = _render_type_block()

    system = prompt["relations_system"].format(relation_types=rel_block)
    user = prompt["relations_user_template"].format(
        source=chunk.get("source", ""), page=chunk.get("page_idx", -1),
        heading=_heading_of(chunk), text=_chunk_text(chunk),
        entities=_render_entity_list(entities))
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _require_dict(obj, original: str) -> dict:
    """顶层必须是 JSON 对象；数组/标量一律视为非法输出。"""
    if not isinstance(obj, dict):
        raise ValueError(f"LLM 输出顶层必须是 JSON 对象（实际 "
                         f"{type(obj).__name__}）：{original[:200]!r}")
    return obj


def parse_llm_json(text: str) -> dict:
    """容错解析 LLM 输出：剥 Markdown 围栏 → 截取首个平衡的 {...}。

    顶层必须是 JSON 对象（dict）；模型若返回数组/标量，直接报错，
    避免调用方在 `payload.get` 上抛出难懂的 AttributeError。
    """
    if not text or not text.strip():
        raise ValueError("LLM 输出为空")

    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return _require_dict(json.loads(cleaned), text)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"LLM 输出中找不到 JSON：{text[:200]!r}")

    depth = 0
    last_error: Exception | None = None
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return _require_dict(json.loads(cleaned[start:i + 1]), text)
                except json.JSONDecodeError as exc:
                    last_error = exc
                    break

    # 严格解析失败（截断/退化重复）→ 抢救完整对象；仍无收获才报错。
    salvaged = salvage_llm_json(cleaned)
    if salvaged["entities"] or salvaged["relations"]:
        return salvaged

    if last_error is not None:
        raise ValueError(f"LLM 输出不是合法 JSON：{last_error}") from last_error
    raise ValueError(f"JSON 括号不闭合：{text[:200]!r}")


_SALVAGE_CAP = 200            # 退化重复输出的回收上限，避免无界循环


def _scan_balanced_objects(text: str, cap: int = _SALVAGE_CAP) -> list[str]:
    """扫描全部**完整闭合**的 {...} 对象原文，尊重字符串字面量与转义。

    与 parse_llm_json 的严格路径不同：截断输出里最外层对象不闭合，但
    其内部嵌套对象（如 entities 数组中的单项）可能已完整，故这里按栈
    匹配每个 `}` 到最近的未闭合 `{`，任意层级都回收。字符串内的
    `{` / `}` / `\\"` 不参与计数。达到 cap 立即停止，防退化重复跑飞。
    """
    objects: list[str] = []
    stack: list[int] = []
    in_str = False
    escaped = False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            objects.append(text[stack.pop():i + 1])
            if len(objects) >= cap:
                break
    return objects


def salvage_llm_json(text: str, cap: int = _SALVAGE_CAP) -> dict:
    """从截断/退化的 LLM 输出中抢救完整对象，返回与严格解析相同的结构。

    逐对象分类：实体（name + type ∈ ENTITY_TYPES）与关系
    （head + tail + type ∈ RELATION_TYPES），其余跳过。产出的仍是原始
    dict，交由 to_entities_relations 施加全部三道闸门——本函数不绕闸门。
    """
    entities: list[dict] = []
    relations: list[dict] = []
    for raw in _scan_balanced_objects(text, cap=cap):
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        etype = obj.get("type")
        name = obj.get("name")
        if (isinstance(etype, str) and etype in ENTITY_TYPES
                and isinstance(name, str) and name.strip()):
            entities.append(obj)
            continue
        head, tail = obj.get("head"), obj.get("tail")
        if (isinstance(etype, str) and etype in RELATION_TYPES
                and isinstance(head, str) and head.strip()
                and isinstance(tail, str) and tail.strip()):
            relations.append(obj)
    return {"entities": entities, "relations": relations}


def _evidence_is_grounded(evidence: str, source_text: str) -> bool:
    """evidence 必须真实出现在原文中（容忍空白差异）。"""
    if not evidence or not evidence.strip():
        return False
    norm = lambda s: "".join(s.split())            # noqa: E731
    return norm(evidence) in norm(source_text)


def to_entities_relations(payload: dict, chunk: dict
                          ) -> tuple[list[Entity], list[Relation]]:
    """LLM 输出 → 图谱对象，并施加三道闸门：

      1. 实体类型必须在本体内（否则丢弃）；
      2. 关系类型必须在本体内，且头尾类型合法（否则丢弃）；
      3. 关系的 evidence 必须真实出现在原文中（否则丢弃）—— 防幻觉。
    """
    source = chunk.get("source", "")
    page = int(chunk.get("page_idx", -1))
    text = chunk.get("text") or ""
    heading = " / ".join(str(h) for h in (chunk.get("heading_path") or []))

    entities: list[Entity] = []
    name_to_id: dict[str, str] = {}

    for raw in payload.get("entities") or []:
        if not isinstance(raw, dict):
            continue
        try:
            name = str(raw.get("name", "")).strip()
            etype = str(raw.get("type", "")).strip()
            if not name or etype not in ENTITY_TYPES:
                continue
            aliases = [str(a) for a in (raw.get("aliases") or [])
                       if str(a).strip()]
            e = Entity(entity_id=f"{etype}:{name}", name=name, type=etype,
                       aliases=aliases,
                       attrs={**(raw.get("attrs") or {}), "source_doc": source,
                              "page": page, "heading": heading})
        except (TypeError, ValueError):
            continue                       # 单项畸形（如 attrs 非 dict）→ 丢弃该项
        if validate_entity(e):
            continue
        entities.append(e)
        name_to_id[name] = e.entity_id
        for a in aliases:
            name_to_id.setdefault(a, e.entity_id)

    relations: list[Relation] = []
    for raw in payload.get("relations") or []:
        if not isinstance(raw, dict):
            continue
        try:
            head = str(raw.get("head", "")).strip()
            tail = str(raw.get("tail", "")).strip()
            rtype = str(raw.get("type", "")).strip()
            evidence = str(raw.get("evidence", "")).strip()
            # 非数值 confidence（如 "high"）视为畸形项 → 丢弃该条，不连坐整块
            conf = float(raw.get("confidence", 0.8) or 0.8)
        except (TypeError, ValueError):
            continue

        if rtype not in RELATION_TYPES:
            continue
        if not _evidence_is_grounded(evidence, text):
            continue                       # 闸门3：证据不落地 → 丢弃
        head_id = name_to_id.get(head)
        tail_id = name_to_id.get(tail)
        if not (head_id and tail_id):
            continue                       # 端点实体未被接受 → 丢弃

        r = Relation(head_id=head_id, rel_type=rtype, tail_id=tail_id,
                     evidence=evidence[:120], source_doc=source, page=page,
                     confidence=min(max(conf, 0.0), 0.99))
        if validate_relation(r):
            continue
        relations.append(r)

    return entities, relations
