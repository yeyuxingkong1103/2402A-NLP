import json
import re
from pathlib import Path

import httpx

from ..config import Settings
from ..models import ModelGateway
from ..storage.vector import MilvusStore
from ..workspace import WorkspaceService
from .plan import CollectionConfig, SearchPlan
from .understand import normalize_article_number

ARTICLE_LOOKUP_EXACT_COLLECTIONS = {"civil_code_articles", "civil_interpretations", "civil_elements"}


ARTICLE_FILTER_FIELDS = {
    "civil_code_articles": "article_number",
    "civil_interpretations": "related_article_numbers_text",
    "civil_cases": "related_article_numbers",
    "civil_elements": "serial_number",
    "civil_evidence": "related_article_numbers",
    "civil_processes": "related_article_numbers",
    "civil_questions": "related_article_numbers",
    "civil_citations": "article_number",
}

LOCAL_RECORD_FIELDS = {
    "civil_code_articles": ("id", "article_content", ("law_name", "chapter")),
    "civil_interpretations": ("id", "content", ("law_name", "document_number", "part_name")),
    "civil_cases": ("case_id", "summary", ("title", "case_number", "cause_of_action", "core_legal_issue")),
    "civil_elements": ("serial_number", "case_summary", ("title", "cause_of_action", "case_number")),
    "civil_evidence": ("evidence_id", "content", ("title", "rule_name", "rule_summary", "source_case_number")),
    "civil_processes": ("process_id", "content", ("title",)),
    "civil_questions": ("question_id", "content", ("question",)),
    "civil_citations": ("citation_id", "content", ("title", "source_id", "target_id")),
}

WEAK_LOCAL_TERMS = {
    "证据",
    "材料",
    "步骤",
    "流程",
    "规则",
    "案件",
    "案例",
    "法院",
    "民法典",
    "中华人民共和国民法典",
    "最高人民法院",
    "引用关系",
}
STRONG_LOCAL_REASONS = {"id_exact", "title_exact", "document_exact", "content_exact", "article_exact"}


def compact_text(value: object) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "")).casefold()


def sequence_value(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


class ProcessedExactStore:
    """从 data/processed 读取结构化 JSON，补足 Milvus 大集合加载失败时的强精确召回。"""

    def __init__(self, processed_dir: str | Path = "data/processed"):
        self.processed_dir = Path(processed_dir or "data/processed")
        self._cache: dict[str, list[dict]] = {}

    def load(self, collection: str) -> list[dict]:
        if collection in self._cache:
            return self._cache[collection]
        path = self.processed_dir / f"{collection}.json"
        if collection not in LOCAL_RECORD_FIELDS or not path.is_file():
            self._cache[collection] = []
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._cache[collection] = []
            return []
        if isinstance(payload, dict):
            for key in ("records", "items", "data", collection):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        rows = payload if isinstance(payload, list) else []
        self._cache[collection] = [row for row in rows if isinstance(row, dict)]
        return self._cache[collection]

    @staticmethod
    def title(row: dict, title_fields: tuple[str, ...], content: str, collection: str) -> str:
        for field in title_fields:
            value = str(row.get(field) or "").strip()
            if value:
                return value
        return content[:80] or collection

    def format_row(self, collection: str, row: dict, score: float, reason: str, priority: set[str]) -> dict:
        primary_field, content_field, title_fields = LOCAL_RECORD_FIELDS[collection]
        content = str(row.get(content_field) or row.get("content") or row.get("summary") or "").strip()
        primary_id = str(row.get(primary_field) or "").strip()
        channel = "keyword" if reason == "keyword_exact" else "exact"
        formatted = {
            "source_id": f"{collection}:{primary_id}",
            "source_type": "public",
            "collection": collection,
            "title": self.title(row, title_fields, content, collection),
            "content": content,
            "score": score,
            "retrieval_channel": channel,
            "retrieval_reason": reason,
            "priority_collection": collection in priority,
            "local_exact_match": True,
            "direct_article_match": collection == "civil_code_articles" and reason == "article_exact",
        }
        metadata_fields = {
            primary_field,
            "article_number",
            "law_name",
            "document_number",
            "part_name",
            "rule_name",
            "rule_summary",
            "question",
            "case_number",
            "cause_of_action",
            "core_legal_issue",
            "source_case_number",
            "target_id",
        }
        for field in metadata_fields:
            value = row.get(field)
            if value not in (None, "", []):
                formatted[field] = value
        return formatted

    @staticmethod
    def article_match(collection: str, row: dict, number: str) -> bool:
        if not number:
            return False
        if collection == "civil_code_articles":
            return str(row.get("article_number") or "") == number
        if collection == "civil_elements":
            return str(row.get("serial_number") or "") == number
        if collection == "civil_citations":
            return str(row.get("article_number") or "") == number
        for field in ("related_article_numbers", "related_article_numbers_text"):
            values = {str(item) for item in sequence_value(row.get(field))}
            if number in values:
                return True
        return False

    @staticmethod
    def own_article_number(collection: str, row: dict) -> str:
        if row.get("article_number") is not None:
            return str(row.get("article_number"))
        if collection == "civil_interpretations":
            match = re.search(r"_(\d+)_(\d+)$", str(row.get("id") or ""))
            if match:
                return match.group(1)
            content_match = re.match(r"第\s*([一二三四五六七八九十百千万零〇两\d]+)\s*条", str(row.get("content") or ""))
            if content_match:
                return normalize_article_number(content_match.group(1))
        return ""

    @staticmethod
    def strong_terms(query: str, keywords: list[str] | None) -> list[str]:
        terms = [query]
        for pattern in (r"“([^”]{4,})”", r"\"([^\"]{4,})\"", r"《([^》]{4,})》"):
            terms.extend(match.group(1) for match in re.finditer(pattern, query))
        terms.extend(keywords or [])
        result = []
        for term in terms:
            compact = compact_text(term)
            if len(compact) < 4 or compact in WEAK_LOCAL_TERMS:
                continue
            if compact not in result:
                result.append(compact)
        return result[:12]

    @staticmethod
    def keyword_terms(query: str, keywords: list[str] | None) -> list[str]:
        """提取适合本地降级检索的关键词，避免 Milvus 不可用时公共库变成空结果。"""
        candidates = list(keywords or [])
        if not candidates:
            candidates = re.split(r"[，。！？；、,:：;\s]+", str(query or ""))
        result = []
        for term in candidates:
            compact = compact_text(term)
            if len(compact) < 2 or compact in WEAK_LOCAL_TERMS:
                continue
            if compact not in result:
                result.append(compact)
        return result[:16]

    def search(
        self,
        query: str,
        collections: list[str],
        *,
        keywords: list[str] | None = None,
        article_numbers: list[str] | None = None,
        priority_collections: list[str] | None = None,
        limit: int = 12,
        include_citations: bool = False,
    ) -> list[dict]:
        query_norm = compact_text(query)
        keyword_terms = self.keyword_terms(query, keywords)
        priority = set(priority_collections or [])
        matches: dict[str, dict] = {}

        def add(collection: str, row: dict, score: float, reason: str) -> None:
            formatted = self.format_row(collection, row, score, reason, priority)
            if collection == "civil_citations" and "引用关系" in query:
                formatted["citation_query_match"] = True
                formatted["score"] = max(float(formatted.get("score", 0) or 0), 0.99)
            key = formatted["source_id"]
            current = matches.get(key)
            if current is None or float(formatted.get("score", 0) or 0) > float(current.get("score", 0) or 0):
                matches[key] = formatted

        for collection in collections:
            if collection == "civil_citations" and not include_citations:
                continue
            if collection not in LOCAL_RECORD_FIELDS:
                continue
            primary_field, content_field, title_fields = LOCAL_RECORD_FIELDS[collection]
            for row in self.load(collection):
                primary_id = str(row.get(primary_field) or "").strip()
                source_id = f"{collection}:{primary_id}"
                primary_norm = compact_text(primary_id)
                source_norm = compact_text(source_id)
                if query_norm and (source_norm and source_norm in query_norm or len(primary_norm) >= 4 and primary_norm in query_norm):
                    add(collection, row, 1.0, "id_exact")
                    continue

                for number in article_numbers or []:
                    if self.article_match(collection, row, str(number)):
                        score = 1.0 if collection == "civil_code_articles" else 0.96
                        add(collection, row, score, "article_exact")

                field_norms = []
                for field in title_fields:
                    if field in {"law_name", "document_number", "chapter"}:
                        continue
                    value = compact_text(row.get(field))
                    if value:
                        field_norms.append(value)
                title_hit = any(value and len(value) >= 6 and value in query_norm for value in field_norms)
                if title_hit:
                    score = 0.99
                    if collection == "civil_interpretations" and str(row.get("article_number") or "") == "1":
                        score = 1.0
                    add(collection, row, score, "title_exact")
                    continue

                if collection == "civil_citations":
                    content_norm = compact_text(row.get(content_field))
                    if content_norm and len(content_norm) >= 8 and (content_norm in query_norm or query_norm in content_norm):
                        add(collection, row, 1.0, "content_exact")
                        continue

                if collection == "civil_interpretations":
                    law_name = compact_text(row.get("law_name"))
                    document_number = compact_text(row.get("document_number"))
                    if law_name and law_name in query_norm and (not document_number or document_number in query_norm):
                        score = 1.0 if self.own_article_number(collection, row) == "1" else 0.86
                        add(collection, row, score, "document_exact")
                        continue

                if keyword_terms:
                    title_values = [
                        compact_text(row.get(field))
                        for field in title_fields
                        if field not in {"law_name", "document_number", "chapter"}
                    ]
                    content_norm = compact_text(row.get(content_field) or row.get("content") or row.get("summary"))
                    title_hits = [term for term in keyword_terms if any(term in value for value in title_values if value)]
                    content_hits = [term for term in keyword_terms if term in content_norm]
                    if title_hits or content_hits:
                        hit_terms = list(dict.fromkeys([*title_hits, *content_hits]))
                        longest = max(map(len, hit_terms), default=2)
                        score = 0.78 + min(0.08, len(hit_terms) * 0.025) + min(0.06, longest * 0.01)
                        if title_hits:
                            score += 0.06
                        add(collection, row, min(0.94, score), "keyword_exact")

        return sorted(
            matches.values(),
            key=lambda row: (
                row.get("direct_article_match") is True,
                row.get("priority_collection") is True,
                float(row.get("score", 0) or 0),
            ),
            reverse=True,
        )[:limit]


class VectorRetriever:
    def __init__(self, model: ModelGateway, milvus: MilvusStore, settings: Settings | None = None):
        self.model = model
        self.milvus = milvus
        self.settings = settings
        processed_dir = getattr(settings, "processed_dir", "") if settings is not None else ""
        self.local_exact = ProcessedExactStore(processed_dir or "data/processed") if settings is not None else None

    @staticmethod
    def source_key(row: dict) -> str:
        return str(row.get("source_id") or row.get("title") or row.get("content", "")[:80])

    @staticmethod
    def article_filter(collection: str, article_number: str) -> str:
        field = ARTICLE_FILTER_FIELDS.get(collection, "")
        number = str(article_number).strip()
        if not field or not number:
            return ""
        if collection == "civil_code_articles":
            return f'id == "civil_code_articles_article_{int(number)}"' if number.isdigit() else ""
        if collection in {"civil_interpretations"}:
            return f'array_contains({field}, "{number}")'
        if collection in {"civil_cases", "civil_evidence", "civil_processes", "civil_questions"}:
            return f"array_contains({field}, {int(number)})" if number.isdigit() else ""
        if collection == "civil_elements":
            return f'{field} == "{number}"'
        return f"{field} == {int(number)}" if number.isdigit() else ""

    def merge_results(self, result_sets: list[list[dict]]) -> list[dict]:
        merged = {}
        for rows in result_sets:
            for row in rows:
                key = self.source_key(row)
                if key not in merged:
                    merged[key] = row
                    continue
                existing = merged[key]
                existing_score = float(existing.get("score", 0) or 0)
                new_score = float(row.get("score", 0) or 0)
                if new_score > existing_score:
                    merged[key] = {**existing, **row}
        return list(merged.values())

    @staticmethod
    def filter_merged_results(rows: list[dict]) -> list[dict]:
        MIN_VECTOR_SCORE = 0.25
        MIN_KEYWORD_SCORE = 0.20
        filtered = []
        for row in rows:
            if row.get("retrieval_channel") == "exact" or row.get("direct_article_match"):
                filtered.append(row)
                continue
            channel = str(row.get("retrieval_channel") or "").strip()
            score = float(row.get("score", 0) or 0)
            if channel == "keyword" and score < MIN_KEYWORD_SCORE:
                continue
            if channel == "vector" and score < MIN_VECTOR_SCORE:
                continue
            filtered.append(row)
        return filtered

    @staticmethod
    def sort_results(rows: list[dict]) -> list[dict]:
        return sorted(
            rows,
            key=lambda row: (
                row.get("direct_article_match") is True,
                row.get("citation_query_match") is True,
                row.get("retrieval_channel") == "exact",
                row.get("priority_collection") is True,
                float(row.get("score", 0) or 0),
            ),
            reverse=True,
        )

    @staticmethod
    def has_strong_local_match(rows: list[dict]) -> bool:
        return any(row.get("local_exact_match") and row.get("retrieval_reason") in STRONG_LOCAL_REASONS for row in rows)

    @staticmethod
    def normalize_keywords(keywords: list[str] | None) -> list[str]:
        normalized = []
        for keyword in keywords or []:
            value = str(keyword or "").strip()
            if value and value not in normalized:
                normalized.append(value)
        return normalized[:8]

    def search_public(
        self,
        query: str,
        collections: list[str],
        limit: int | None = None,
        query_variants: list[str] | None = None,
        priority_collections: list[str] | None = None,
        article_numbers: list[str] | None = None,
        keywords: list[str] | None = None,
        wants_citation_relations: bool = False,
        collection_configs: dict[str, CollectionConfig] | None = None,
    ) -> list[dict]:
        """使用多查询、多库权重和条号过滤补强公共法律知识库召回。"""
        base_limit = limit or setting_value(self.settings, "retrieval_vector_top_k", 4)
        priority_limit = setting_value(self.settings, "retrieval_priority_top_k", base_limit + 2)
        non_priority_limit = setting_value(self.settings, "retrieval_non_priority_top_k", max(1, base_limit // 2))
        keyword_limit = setting_value(self.settings, "retrieval_keyword_top_k", base_limit)
        exact_limit = setting_value(self.settings, "retrieval_exact_top_k", max(base_limit, 5))
        candidate_limit = setting_value(self.settings, "retrieval_candidate_pool_max", 80)
        max_variants = setting_value(self.settings, "retrieval_max_query_variants", 4)
        non_priority_variants = setting_value(self.settings, "retrieval_non_priority_query_variants", 1)
        max_collections = setting_value(self.settings, "retrieval_max_collections", len(collections) or 1)

        variants = list(dict.fromkeys([query, *(query_variants or [])]))[:max_variants]
        priority = set(priority_collections or [])
        ordered_collections = [collection for collection in priority_collections or [] if collection in collections]
        ordered_collections.extend(collection for collection in collections if collection not in ordered_collections)
        vector_collections = ordered_collections[:max_collections]
        normalized_keywords = self.normalize_keywords(keywords)
        citation_relations = bool(wants_citation_relations)
        if article_numbers and not citation_relations:
            exact_collections = [collection for collection in ordered_collections if collection in ARTICLE_LOOKUP_EXACT_COLLECTIONS]
        else:
            exact_collections = ordered_collections
        collection_configs = {
            collection: (collection_configs or {}).get(collection, CollectionConfig())
            for collection in ordered_collections
        }
        result_sets: list[list[dict]] = []
        local_rows: list[dict] = []
        milvus_failed = False

        if self.local_exact is not None:
            local_rows = self.local_exact.search(
                query,
                ordered_collections,
                keywords=normalized_keywords,
                article_numbers=article_numbers,
                priority_collections=priority_collections,
                limit=candidate_limit,
                include_citations=citation_relations,
            )
            if local_rows:
                result_sets.append(local_rows)
                if self.has_strong_local_match(local_rows):
                    return self.sort_results(self.merge_results(result_sets))[:candidate_limit]

        weak_query = len(str(query or "").strip()) <= 2
        if weak_query and not normalized_keywords and not article_numbers and all(len(str(variant or "").strip()) <= 2 for variant in variants):
            return self.sort_results(self.merge_results(result_sets))[:candidate_limit] if result_sets else []

        for number in article_numbers or []:
            for collection in exact_collections:
                config = collection_configs.get(collection, CollectionConfig())
                if not config.enable_exact:
                    continue
                filter_expr = self.article_filter(collection, number)
                if not filter_expr:
                    continue
                try:
                    if hasattr(self.milvus, "query_public"):
                        exact_score = 1.0 if collection == "civil_code_articles" else 0.95
                        rows = self.milvus.query_public(collection, filter_expr, min(exact_limit, config.top_k), score=exact_score)
                    else:
                        rows = []
                except Exception:
                    milvus_failed = True
                    rows = []
                for row in rows:
                    row["query_variant"] = f"民法典第{number}条"
                    row["retrieval_channel"] = "exact"
                    row["retrieval_reason"] = "article_exact"
                    row["priority_collection"] = collection in priority
                    row["direct_article_match"] = collection == "civil_code_articles"
                rows = [row for row in rows if float(row.get("score", 0) or 0) >= config.min_score or row.get("direct_article_match")]
                result_sets.append(rows)

        if any(row.get("direct_article_match") for rows in result_sets for row in rows):
            return self.sort_results(self.merge_results(result_sets))[:candidate_limit]

        if milvus_failed and local_rows:
            return self.sort_results(self.merge_results(result_sets))[:candidate_limit]

        if normalized_keywords:
            keyword_text = " ".join(normalized_keywords)
            for collection in vector_collections:
                config = collection_configs.get(collection, CollectionConfig())
                if not config.enable_keyword:
                    continue
                try:
                    rows = self.milvus.search_public_keyword(collection, normalized_keywords, min(keyword_limit, config.top_k))
                except Exception:
                    milvus_failed = True
                    rows = []
                for row in rows:
                    row["query_variant"] = keyword_text
                    row["retrieval_channel"] = "keyword"
                    row["retrieval_reason"] = "keyword"
                    row["keywords"] = normalized_keywords
                    row["priority_collection"] = collection in priority
                rows = [row for row in rows if float(row.get("score", 0) or 0) >= config.min_score]
                result_sets.append(rows)
                if milvus_failed and local_rows:
                    return self.sort_results(self.merge_results(result_sets))[:candidate_limit]

        needed_variants = []
        for collection in vector_collections:
            collection_variants = variants if collection in priority else variants[:non_priority_variants]
            needed_variants.extend(collection_variants)
        needed_variants = list(dict.fromkeys(needed_variants))
        vectors = []
        vector_by_variant = {}
        if needed_variants:
            try:
                vectors = self.model.embed(needed_variants)
                vector_by_variant = dict(zip(needed_variants, vectors))
            except Exception:
                milvus_failed = True
                vectors = []
                vector_by_variant = {}
        if milvus_failed and local_rows and not vector_by_variant:
            return self.sort_results(self.merge_results(result_sets))[:candidate_limit]
        for collection in vector_collections:
            config = collection_configs.get(collection, CollectionConfig())
            if not config.enable_vector:
                continue
            collection_variants = variants if collection in priority else variants[:non_priority_variants]
            collection_limit = min(config.top_k, priority_limit if collection in priority else non_priority_limit)
            for variant in collection_variants:
                vector = vector_by_variant.get(variant)
                if vector is None:
                    continue
                try:
                    rows = self.milvus.search_public(collection, vector, collection_limit)
                except Exception:
                    milvus_failed = True
                    rows = []
                for row in rows:
                    row["query_variant"] = variant
                    row["retrieval_channel"] = "vector"
                    row["priority_collection"] = collection in priority
                    if collection in priority:
                        row["retrieval_reason"] = "priority_vector"
                    else:
                        row["retrieval_reason"] = "vector"
                rows = [row for row in rows if float(row.get("score", 0) or 0) >= config.min_score]
                result_sets.append(rows)
                if milvus_failed and local_rows:
                    return self.sort_results(self.merge_results(result_sets))[:candidate_limit]
        if not hasattr(self.milvus, "query_public") and vector_by_variant:
            base_vector = vector_by_variant.get(query) or next(iter(vector_by_variant.values()))
            for number in article_numbers or []:
                for collection in exact_collections:
                    filter_expr = self.article_filter(collection, number)
                    if not filter_expr:
                        continue
                    try:
                        rows = self.milvus.search_public(collection, base_vector, exact_limit, filter_expr=filter_expr)
                    except Exception:
                        rows = []
                    for row in rows:
                        row["query_variant"] = f"民法典第{number}条"
                        row["retrieval_channel"] = "exact"
                        row["retrieval_reason"] = "article_exact"
                        row["priority_collection"] = collection in priority
                        row["direct_article_match"] = collection == "civil_code_articles"
                    result_sets.append(rows)

        merged = self.filter_merged_results(self.merge_results(result_sets))
        return self.sort_results(merged)[:candidate_limit]


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def mojibake_ratio(text: str) -> float:
    value = str(text or "")
    if not value:
        return 0.0
    bad = sum(1 for char in value if char in "�□▯�" or "\ud800" <= char <= "\udfff")
    bars = value.count("|")
    return (bad + min(bars, len(value) // 3)) / max(1, len(value))


def clean_web_text(text: str, limit: int = 3000) -> str:
    value = str(text or "")
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", value)
    value = re.sub(r"\[[^\]]{0,30}\]\([^)]*\)", " ", value)
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"[|]{3,}", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if mojibake_ratio(value) > 0.08:
        return ""
    return value[:limit]


def setting_value(settings: Settings | None, name: str, default: int) -> int:
    value = getattr(settings, name, default) if settings is not None else default
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


class WebRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_error = ""

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        if not self.settings.tavily_api_key:
            return []
        max_results = limit or self.settings.tavily_max_results
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": self.settings.tavily_search_depth,
            "include_raw_content": True,
        }
        allowed_domains = csv_values(self.settings.tavily_allowed_domains)
        blocked_domains = csv_values(self.settings.tavily_blocked_domains)
        if allowed_domains:
            payload["include_domains"] = allowed_domains
        if blocked_domains:
            payload["exclude_domains"] = blocked_domains
        try:
            response = httpx.post(
                f"{self.settings.tavily_base_url.rstrip('/')}/search",
                json=payload,
                timeout=self.settings.tavily_timeout,
            )
            response.raise_for_status()
            items = response.json().get("results", [])
        except Exception as exc:
            self.last_error = str(exc)
            return []
        trusted_domains = csv_values(self.settings.tavily_trusted_domains)
        results = []
        for item in items:
            url = str(item.get("url", ""))
            title = clean_web_text(item.get("title", ""), limit=180) or "网页资料"
            content = clean_web_text(item.get("raw_content") or item.get("content", ""))
            score = float(item.get("score", 0) or 0)
            if trusted_domains and any(domain in url for domain in trusted_domains):
                score = min(1.0, score + 0.1)
            row = {
                "source_id": url,
                "source_type": "web",
                "title": title,
                "content": content,
                "score": score,
            }
            if row.get("content") or row.get("title"):
                results.append(row)
        return results


class Retriever:
    def __init__(self, settings: Settings, model: ModelGateway, milvus: MilvusStore, workspace: WorkspaceService):
        self.settings = settings
        self.public = VectorRetriever(model, milvus, settings)
        self.workspace = workspace
        self.web = WebRetriever(settings)

    def search_public_knowledge(self, plan: SearchPlan) -> list[dict]:
        """统一检索公共民事法律知识库，按计划执行多查询和重点库召回。"""
        if not plan.search_public:
            return []
        search = getattr(self.public, "search_public", None)
        if search is None:
            search = getattr(self.public, "search", None)
        if search is None:
            return []
        try:
            return search(
                plan.query,
                plan.collections,
                query_variants=plan.query_variants,
                priority_collections=plan.priority_collections,
                article_numbers=plan.article_numbers,
                keywords=plan.keywords,
                wants_citation_relations=plan.wants_citation_relations,
                collection_configs=getattr(plan, "collection_configs", None),
            )
        except TypeError:
            try:
                return search(plan.query, plan.collections)
            except TypeError:
                return search(plan.query, plan.collections, collection_configs=getattr(plan, "collection_configs", None))

    def search_user_materials(self, plan: SearchPlan, user: dict | None, session_id: str | None) -> list[dict]:
        """检索当前用户上传的私有材料。未登录时不查私有材料。"""
        if not user or not plan.search_private:
            return []
        settings = getattr(self, "settings", None)
        parts = [
            plan.query,
            plan.original_query,
            plan.rewritten_query,
            *(plan.query_variants or []),
            *(plan.keywords or []),
        ]
        query_parts = []
        for part in parts:
            value = str(part or "").strip()
            if value and value not in query_parts:
                query_parts.append(value)
        query = "\n".join(query_parts) or plan.query
        try:
            rows = self.workspace.search(user, query, session_id, limit=setting_value(settings, "retrieval_private_top_k", 6))
        except TypeError:
            rows = self.workspace.search(user, query, session_id)

        seen = set()
        unique_rows = []
        for row in rows:
            key = f"{row.get('source_id', '')}:{row.get('title', '')}"
            if key in seen:
                continue
            seen.add(key)
            row.setdefault("retrieval_channel", "private")
            unique_rows.append(row)

        return unique_rows

    def search_web_pages(self, plan: SearchPlan) -> list[dict]:
        """按用户设置决定是否联网检索。"""
        if not plan.include_web:
            return []
        settings = getattr(self, "settings", None)
        try:
            rows = self.web.search(plan.query, setting_value(settings, "retrieval_web_top_k", 5))
        except TypeError:
            rows = self.web.search(plan.query)
        for row in rows:
            row.setdefault("retrieval_channel", "web")
        return rows

    def search(self, plan: SearchPlan, user: dict | None, session_id: str | None = None) -> list[dict]:
        results = []
        results.extend(self.search_user_materials(plan, user, session_id))
        results.extend(self.search_public_knowledge(plan))
        results.extend(self.search_web_pages(plan))
        return results


from ..models import ModelGateway


PRIVATE_SOURCE_TYPES = {"private", "user_upload"}
CORE_PUBLIC_COLLECTIONS = {"civil_code_articles", "civil_interpretations", "civil_elements"}
MIN_PUBLIC_RELEVANCE_SCORE = 0.55
MIN_PRIORITY_REPRESENTATIVE_SCORE = 0.55  # 从 0.65 改为 0.55
DEFAULT_CHANNEL_WEIGHTS = {
    "exact": 2.4,
    "private": 2.0,
    "keyword": 1.35,
    "vector": 1.0,
    "web": 0.7,
    "unknown": 1.0,
}
CHANNEL_PRIORITY = {"exact": 5, "private": 4, "keyword": 3, "vector": 2, "web": 1, "unknown": 0}


def source_key(row: dict) -> str:
    return str(row.get("source_id") or row.get("title") or row.get("content", "")[:50])


def retrieval_channel(item: dict) -> str:
    channel = str(item.get("retrieval_channel") or "").strip()
    if channel:
        return channel
    if item.get("retrieval_reason") == "article_exact":
        return "exact"
    if item.get("source_type") in PRIVATE_SOURCE_TYPES:
        return "private"
    if item.get("source_type") == "web":
        return "web"
    return "unknown"


def relevance_score(row: dict) -> float:
    scores = []
    for key in ("rerank_score", "score", "rrf_score"):
        value = row.get(key)
        try:
            scores.append(float(value or 0))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else 0.0


def retrieval_score(row: dict) -> float:
    value = row.get("score", row.get("rrf_score", 0))
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def is_exact_match(row: dict) -> bool:
    return retrieval_channel(row) == "exact" or row.get("retrieval_reason") == "article_exact"


def source_priority(row: dict) -> int:
    channel = retrieval_channel(row)
    if row.get("direct_article_match"):
        return 7
    if row.get("citation_query_match"):
        return 6
    if channel == "exact" and row.get("collection") in CORE_PUBLIC_COLLECTIONS:
        return 6
    if channel == "exact" or row.get("retrieval_reason") == "article_exact":
        return 5
    if row.get("source_type") in PRIVATE_SOURCE_TYPES:
        return 4
    if channel == "keyword":
        return 3
    if row.get("priority_collection") or row.get("collection") in CORE_PUBLIC_COLLECTIONS:
        return 2
    if row.get("source_type") == "public":
        return 1
    return 0


def rrf(
    channels: list[list[dict]],
    limit: int = 12,
    k: int = 60,
    channel_weights: dict[str, float] | None = None,
    private_limit: int = 4,
    exact_limit: int = 4,
) -> list[dict]:
    weights = {**DEFAULT_CHANNEL_WEIGHTS, **(channel_weights or {})}
    merged: dict[str, dict] = {}
    for rows in channels:
        for rank, item in enumerate(rows, start=1):
            key = source_key(item)
            channel = retrieval_channel(item)
            current = merged.setdefault(key, {**item, "source_id": key, "rrf_score": 0.0, "matched_channels": []})
            weight = float(weights.get(channel, weights.get("unknown", 1.0)))
            if channel == "vector" and item.get("priority_collection"):
                weight *= 1.15
            if item.get("collection") in CORE_PUBLIC_COLLECTIONS:
                weight *= 1.05
            current["rrf_score"] += weight / (k + rank)
            if channel not in current["matched_channels"]:
                current["matched_channels"].append(channel)
            current_channel = retrieval_channel(current)
            if CHANNEL_PRIORITY.get(channel, 0) > CHANNEL_PRIORITY.get(current_channel, 0):
                preserved = {"rrf_score": current["rrf_score"], "matched_channels": current["matched_channels"]}
                current.update(item)
                current.update(preserved)
            elif relevance_score(item) > relevance_score(current):
                current["score"] = item.get("score", current.get("score"))
    ranked = sorted(
        merged.values(),
        key=lambda row: (source_priority(row), row["rrf_score"], relevance_score(row)),
        reverse=True,
    )
    selected = []
    selected_keys = set()

    exact_rows = [item for item in ranked if retrieval_channel(item) == "exact" or item.get("retrieval_reason") == "article_exact"]
    for item in exact_rows[:exact_limit]:
        if len(selected) >= limit:
            break
        if item["source_id"] not in selected_keys:
            selected.append(item)
            selected_keys.add(item["source_id"])

    private_rows = [item for item in ranked if item.get("source_type") in PRIVATE_SOURCE_TYPES]
    public_exists = any(item.get("source_type") == "public" for item in ranked)
    private_quota = min(limit - len(selected), private_limit if public_exists else limit)
    for item in private_rows[:private_quota]:
        if len(selected) >= limit:
            break
        if item["source_id"] not in selected_keys:
            selected.append(item)
            selected_keys.add(item["source_id"])

    for item in ranked:
        if len(selected) >= limit:
            break
        if item["source_id"] in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(item["source_id"])

    return selected


class Reranker:
    def __init__(self, model: ModelGateway, input_limit: int = 32, doc_max_chars: int = 1200):
        self.model = model
        self.input_limit = max(1, int(input_limit or 32))
        self.doc_max_chars = max(200, int(doc_max_chars or 1200))

    def keep_private_materials(self, original_rows: list[dict], ranked_rows: list[dict]) -> list[dict]:
        """用户上传材料不能在重排后消失，否则回答会忽略用户自己的证据。"""
        existing_keys = {source_key(row) for row in ranked_rows}
        private_rows = [
            row for row in original_rows
            if row.get("source_type") in PRIVATE_SOURCE_TYPES and source_key(row) not in existing_keys
        ]
        return private_rows + ranked_rows

    def keep_exact_matches(self, original_rows: list[dict], ranked_rows: list[dict]) -> list[dict]:
        existing_keys = {source_key(row) for row in ranked_rows}
        exact_rows = [
            row for row in original_rows
            if is_exact_match(row)
            and source_key(row) not in existing_keys
        ]
        return sorted(exact_rows, key=lambda row: (source_priority(row), retrieval_score(row)), reverse=True) + ranked_rows

    def exact_matches(self, rows: list[dict]) -> list[dict]:
        exact_rows = [row for row in rows if is_exact_match(row)]
        return sorted(exact_rows, key=lambda row: (source_priority(row), retrieval_score(row)), reverse=True)

    def keep_public_library_representatives(self, original_rows: list[dict], ranked_rows: list[dict]) -> list[dict]:
        """只补回确实相关的重点/核心公共库代表，避免低相关类案或模板混入答案。"""
        existing_collections = {
            row.get("collection")
            for row in ranked_rows
            if row.get("source_type") == "public"
        }
        existing_keys = {source_key(row) for row in ranked_rows}
        missing_rows = []
        priority_collections = []
        priority_candidates = sorted(
            original_rows,
            key=lambda row: (row.get("priority_collection") is True, source_priority(row), relevance_score(row)),
            reverse=True,
        )
        for row in priority_candidates:
            collection = row.get("collection")
            if (
                row.get("source_type") == "public"
                and (row.get("priority_collection") or collection in CORE_PUBLIC_COLLECTIONS)
                and collection not in priority_collections
                and (is_exact_match(row) or relevance_score(row) >= MIN_PRIORITY_REPRESENTATIVE_SCORE)
            ):
                priority_collections.append(collection)
        for collection in priority_collections[:3]:
            if collection in existing_collections:
                continue
            match = next((
                row for row in original_rows
                if row.get("source_type") == "public"
                and row.get("collection") == collection
                and source_key(row) not in existing_keys
                and (is_exact_match(row) or relevance_score(row) >= MIN_PRIORITY_REPRESENTATIVE_SCORE)
                # ========== 新增：额外检查分数 ==========
                and relevance_score(row) >= MIN_PRIORITY_REPRESENTATIVE_SCORE
                # ========== 额外检查结束 ==========
            ), None)
            if match:
                missing_rows.append(match)
                existing_keys.add(source_key(match))
        return ranked_rows + missing_rows

    def rank(self, query: str, rows: list[dict]) -> list[dict]:
        return self.rerank(query, rows)

    def rerank(self, query: str, rows: list[dict]) -> list[dict]:
        exact_rows = self.exact_matches(rows)
        if exact_rows:
            exact_keys = {source_key(row) for row in exact_rows}
            remaining_rows = [row for row in rows if source_key(row) not in exact_keys]
            ranked = exact_rows + remaining_rows
            ranked = self.keep_public_library_representatives(rows, ranked)
            ranked = self.keep_private_materials(rows, ranked)
            return ranked

        # ========== 新增：重排前过滤低分文档 ==========
        MIN_INPUT_SCORE = 0.15
        filtered_rows = []
        for row in rows[:self.input_limit]:
            score = float(row.get("rrf_score", row.get("score", 0)) or 0)
            # 私有材料永远保留
            if score >= MIN_INPUT_SCORE or row.get("source_type") in PRIVATE_SOURCE_TYPES or ("score" not in row and "rrf_score" not in row):
                filtered_rows.append(row)

        # 如果过滤后为空，返回前 8 条（降级策略）
        if not filtered_rows:
            return rows[:min(8, len(rows))]
        # ========== 过滤结束 ==========

        rows_for_model = filtered_rows[:self.input_limit]
        docs = [item.get("content", "")[:self.doc_max_chars] for item in rows_for_model]
        ranked = []
        try:
            pairs = self.model.rerank(query, docs)
        except Exception:
            return self.keep_exact_matches(rows, self.keep_private_materials(rows, rows))
        for index, score in pairs:
            if 0 <= index < len(rows_for_model):
                ranked.append({**rows_for_model[index], "rerank_score": score})
        ranked = self.keep_public_library_representatives(rows, ranked or rows)
        ranked = self.keep_private_materials(rows, ranked)
        return self.keep_exact_matches(rows, ranked)


class EvidenceBuilder:
    def __init__(self, limit: int = 8, max_content_chars: int = 1200, private_limit: int = 4, exact_limit: int = 4):
        self.limit = max(1, int(limit or 8))
        self.max_content_chars = max(200, int(max_content_chars or 1200))
        self.private_limit = max(1, int(private_limit or 4))
        self.exact_limit = max(1, int(exact_limit or 4))

    @staticmethod
    def relevance_score(row: dict) -> float:
        scores = []
        for key in ("rerank_score", "score", "rrf_score"):
            value = row.get(key)
            try:
                scores.append(float(value or 0))
            except (TypeError, ValueError):
                continue
        return max(scores) if scores else 0.0

    def clean_rows(self, rows: list[dict]) -> list[dict]:
        cleaned = []
        seen = set()
        for row in rows:
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            source_id = source_key(row)
            if source_id in seen:
                continue
            seen.add(source_id)
            cleaned.append({**row, "source_id": source_id, "content": content[:self.max_content_chars]})
        return cleaned

    def add_unique(self, evidence: list[dict], used: set[str], row: dict) -> None:
        key = row["source_id"]
        if key in used:
            return
        evidence.append(row)
        used.add(key)

    def build(self, rows: list[dict]) -> list[dict]:
        cleaned = self.clean_rows(rows)
        evidence = []
        used = set()

        private_rows = [row for row in cleaned if row.get("source_type") in PRIVATE_SOURCE_TYPES]
        for row in sorted(private_rows, key=self.relevance_score, reverse=True)[:self.private_limit]:
            self.add_unique(evidence, used, row)

        public_rows = [
            row for row in cleaned
            if row.get("source_type") == "public"
            and (row.get("collection") != "civil_citations" or row.get("citation_query_match"))
            and (is_exact_match(row) or relevance_score(row) >= MIN_PUBLIC_RELEVANCE_SCORE)
        ]
        exact_rows = [row for row in public_rows if row.get("retrieval_reason") == "article_exact" or retrieval_channel(row) == "exact"]
        for row in sorted(exact_rows, key=lambda row: (source_priority(row), self.relevance_score(row)), reverse=True)[:self.exact_limit]:
            if len(evidence) >= self.limit:
                break
            self.add_unique(evidence, used, row)

        priority_rows = [row for row in public_rows if row.get("priority_collection") or row.get("collection") in CORE_PUBLIC_COLLECTIONS]
        for row in sorted(priority_rows, key=lambda row: (source_priority(row), self.relevance_score(row)), reverse=True):
            if len(evidence) >= self.limit:
                break
            self.add_unique(evidence, used, row)

        remaining = sorted(
            [
                row for row in cleaned
                if row["source_id"] not in used
                and (
                    row.get("source_type") not in {"public", "web"}
                    or (row.get("collection") != "civil_citations" or row.get("citation_query_match"))
                    and (is_exact_match(row) or relevance_score(row) >= MIN_PUBLIC_RELEVANCE_SCORE)
                )
            ],
            key=lambda row: (
                source_priority(row),
                self.relevance_score(row),
            ),
            reverse=True,
        )
        for row in remaining:
            if len(evidence) >= self.limit:
                break
            self.add_unique(evidence, used, row)
        return evidence


class ProcessingPipeline:
    def __init__(self, reranker: Reranker, evidence: EvidenceBuilder, rrf_limit: int = 12, rrf_k: int = 60, channel_weights: dict[str, float] | None = None):
        self.ranker = reranker
        self.evidence = evidence
        self.rrf_limit = max(1, int(rrf_limit or 12))
        self.rrf_k = max(1, int(rrf_k or 60))
        self.channel_weights = channel_weights or None

    def merge_results(self, channels: list[list[dict]]) -> list[dict]:
        return rrf(channels, limit=self.rrf_limit, k=self.rrf_k, channel_weights=self.channel_weights)

    def rank(self, query: str, rows: list[dict]) -> list[dict]:
        return self.ranker.rerank(query, rows)

    def build_evidence(self, rows: list[dict]) -> list[dict]:
        return self.evidence.build(rows)

    def process(self, query: str, channels: list[list[dict]]) -> dict:
        fused = self.merge_results(channels)
        ranked = self.rank(query, fused)
        evidence = self.build_evidence(ranked)
        return {
            "fused": fused,
            "ranked": ranked,
            "evidence": evidence,
            "counts": {
                "fused": len(fused),
                "ranked": len(ranked),
                "evidence": len(evidence),
            },
        }

