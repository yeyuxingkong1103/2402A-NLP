from dataclasses import dataclass
import re


@dataclass(frozen=True)
class QueryAnalysis:
    original: str
    normalized: str
    intent: str
    entities: tuple[str, ...]
    rewrites: tuple[str, ...]


def normalize_query(query: str) -> str:
    query = re.sub(r"\s+", " ", query.strip())
    return query.rstrip("？?。！!")


def detect_intent(query: str) -> str:
    if re.search(r"多少|几|比例|增长率|金额|收入|利润|数量", query):
        return "数值查询"
    if re.search(r"比较|相比|差异|排名", query):
        return "比较查询"
    if re.search(r"为什么|原因|影响|风险", query):
        return "原因分析"
    if re.search(r"是否|有没有|能否|可以", query):
        return "判断查询"
    return "事实查询"


def extract_entities(query: str) -> tuple[str, ...]:
    cleaned = re.sub(r"^(请问|请|帮我|想了解)", "", query)
    cleaned = re.sub(r"^(?:本公司|该公司|公司)的?", "", cleaned)
    cleaned = re.sub(r"(是多少|是什么|有哪些|有多少|如何|为什么|吗)$", "", cleaned)
    candidates = re.findall(r"[一-鿿A-Za-z0-9]{2,}", cleaned)
    stopwords = {"公司", "公司的", "报告中", "这个", "具体数值"}
    return tuple(dict.fromkeys(item for item in candidates if item not in stopwords))


def analyze_query(query: str) -> QueryAnalysis:
    normalized = normalize_query(query)
    entities = extract_entities(normalized)
    rewrites = [normalized]
    if entities:
        rewrites.append(" ".join(entities))
    if "多少" in normalized and "数值" not in normalized:
        rewrites.append(normalized.replace("多少", "具体数值"))
    return QueryAnalysis(query, normalized, detect_intent(normalized), entities, tuple(dict.fromkeys(rewrites)))
