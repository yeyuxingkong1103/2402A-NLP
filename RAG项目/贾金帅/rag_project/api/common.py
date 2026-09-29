"""Small response and retrieval-result helpers shared by routes."""
import json
import re


def _encode_stream_event(event: dict) -> str:  # 把流事件编码成一行 JSON
    """将单个流事件编码为一行 JSON，便于浏览器增量解析。"""
    return json.dumps(event, ensure_ascii=False) + "\n"  # 中文不转义 + 换行分隔


def _extract_references(fused_results) -> list[dict]:
    """从融合结果提取引用：来源名 / 溯源链接 / 证据等级。"""
    references: list[dict] = []  # 引用列表
    seen: set = set()  # 去重
    for result in fused_results:  # 遍历融合结果
        content = getattr(result, "content", None) or {}  # 结果内容
        if not isinstance(content, dict):  # 不是字典就跳过
            continue
        name = content.get("name") or content.get("title") or ""  # 来源名
        text = str(content.get("document") or content.get("text") or "")  # 块正文
        source_path = content.get("source_path") or ""  # 文件路径（兜底）
        url_match = re.search(r"\*\*来源链接\*\*[:：]?\s*(https?://\S+)", text)  # 溯源链接
        level_match = re.search(r"\*\*证据等级\*\*[:：]?\s*([高中低])", text)  # 证据等级
        url = url_match.group(1).strip().rstrip("，。,.;；") if url_match else ""  # URL
        if not url:
            url = str(content.get("url") or "").strip()  # Tavily 结果直接提供的 URL
        level = level_match.group(1) if level_match else ""  # 高/中
        text_preview = text[:1500]  # 命中正文（用于前端"伪链接"查看）
        section_path = content.get("section_path") or ""  # 章节路径
        if isinstance(section_path, list):  # 列表转成路径串
            section_path = " > ".join(str(item) for item in section_path)
        key = url or (name + source_path)  # 去重键
        if not key or key in seen:  # 空或重复
            continue
        seen.add(key)  # 标记已见
        references.append({  # 组装引用
            "name": name or source_path or "来源",  # 来源名
            "url": url,  # 溯源链接（可能为空）
            "evidence_level": level,  # 证据等级
            "source_path": source_path,  # 源文件路径
            "section_path": str(section_path),  # 章节路径
            "text": text_preview,  # 命中正文（前端查看用）
        })
    return references[:10]  # 最多 10 条


def _determine_evidence_level(result) -> str:
    """根据来源类型 / 文件路径判定证据等级（指南/试验=高，文献/说明书=中）。"""
    content = result.content or {}  # 结果内容
    source = getattr(result, "source", "") or ""  # 来源引擎
    title = str(content.get("title") or "").lower()  # 标题（小写）
    source_path = str(content.get("source_path") or "").lower()  # 文件路径（小写）
    text = str(content.get("document") or content.get("text") or "")  # 正文
    meta = re.search(r"\*\*证据等级\*\*[:：]?\s*([高中低])", text)  # 元信息行优先
    if meta:  # 数据自带证据等级
        return meta.group(1)
    if source == "graph":  # 图谱=中
        return "中"
    # vector：按路径关键词判定
    if "指南" in title or "guideline" in source_path:
        return "高"
    if "chictr" in source_path or "试验" in source_path:
        return "高"
    if "wanfang" in source_path or "文献" in source_path:
        return "中"
    if ".pdf" in source_path:
        return "中"
    return "中"
