import copy

import pytest

from backend.app.storage.public_schema import (
    PublicDatabaseMismatch,
    validate_public_database,
)
from backend.app.storage.vector import MilvusStore


EXPECTED_COUNTS = {
    "civil_code_articles": 1260,
    "civil_interpretations": 1014,
    "civil_cases": 2980,
    "civil_elements": 1260,
    "civil_evidence": 2460,
    "civil_processes": 448,
    "civil_questions": 18832,
    "civil_citations": 12734,
}

PRIMARY_KEYS = {
    "civil_code_articles": "id",
    "civil_interpretations": "id",
    "civil_cases": "case_id",
    "civil_elements": "serial_number",
    "civil_evidence": "evidence_id",
    "civil_processes": "process_id",
    "civil_questions": "question_id",
    "civil_citations": "citation_id",
}


class FakeMilvusClient:
    def __init__(self):
        self.collections = list(EXPECTED_COUNTS)
        self.schemas = {
            name: {
                "collection_name": name,
                "auto_id": False,
                "num_shards": 1,
                "description": "",
                "fields": [
                    {
                        "field_id": 100,
                        "name": primary_key,
                        "description": "",
                        "type": 21,
                        "params": {"max_length": 128},
                        "is_primary": True,
                    },
                    {
                        "field_id": 101,
                        "name": "embedding",
                        "description": "",
                        "type": 101,
                        "params": {"dim": 1024},
                    },
                ],
                "functions": [],
                "aliases": [],
                "consistency_level": 2,
                "properties": {},
                "num_partitions": 1,
                "enable_dynamic_field": True,
            }
            for name, primary_key in PRIMARY_KEYS.items()
        }
        self.counts = copy.deepcopy(EXPECTED_COUNTS)

    def list_collections(self):
        return list(self.collections)

    def describe_collection(self, collection_name):
        return copy.deepcopy(self.schemas[collection_name])

    def get_collection_stats(self, collection_name):
        return {"row_count": self.counts[collection_name]}


def test_valid_database_returns_all_public_collection_counts():
    client = FakeMilvusClient()

    counts = validate_public_database(client, expected_dimension=1024)

    assert counts == {
        "civil_code_articles": 1260,
        "civil_interpretations": 1014,
        "civil_cases": 2980,
        "civil_elements": 1260,
        "civil_evidence": 2460,
        "civil_processes": 448,
        "civil_questions": 18832,
        "civil_citations": 12734,
    }


def test_missing_collection_is_rejected():
    client = FakeMilvusClient()
    client.collections.remove("civil_questions")

    with pytest.raises(PublicDatabaseMismatch, match="civil_questions"):
        validate_public_database(client)


def test_unexpected_deprecated_collection_is_rejected():
    client = FakeMilvusClient()
    client.collections.append("legal_cases")

    with pytest.raises(PublicDatabaseMismatch, match="legal_cases"):
        validate_public_database(client)


def test_wrong_vector_field_is_rejected():
    client = FakeMilvusClient()
    client.schemas["civil_cases"]["fields"][1]["name"] = "vector"

    with pytest.raises(PublicDatabaseMismatch, match="civil_cases.*embedding"):
        validate_public_database(client)


def test_wrong_vector_dimension_is_rejected():
    client = FakeMilvusClient()
    client.schemas["civil_evidence"]["fields"][1]["params"]["dim"] = 512

    with pytest.raises(PublicDatabaseMismatch, match="civil_evidence.*1024"):
        validate_public_database(client)


def test_wrong_primary_key_is_rejected():
    client = FakeMilvusClient()
    client.schemas["civil_processes"]["fields"][0]["is_primary"] = False

    with pytest.raises(PublicDatabaseMismatch, match="civil_processes.*process_id"):
        validate_public_database(client)


def test_health_reports_counts_cached_by_startup_validation():
    store = MilvusStore.__new__(MilvusStore)
    store.public_collection_counts = {
        "civil_code_articles": 1260,
        "civil_interpretations": 1014,
        "civil_cases": 2980,
        "civil_elements": 1260,
        "civil_evidence": 2460,
        "civil_processes": 448,
        "civil_questions": 18832,
        "civil_citations": 12734,
    }

    health = store.health()

    assert health == {
        "connected": True,
        "public_collections": [
            "civil_code_articles",
            "civil_interpretations",
            "civil_cases",
            "civil_elements",
            "civil_evidence",
            "civil_processes",
            "civil_questions",
            "civil_citations",
        ],
        "public_collection_counts": {
            "civil_code_articles": 1260,
            "civil_interpretations": 1014,
            "civil_cases": 2980,
            "civil_elements": 1260,
            "civil_evidence": 2460,
            "civil_processes": 448,
            "civil_questions": 18832,
            "civil_citations": 12734,
        },
        "indexed_records": 40988,
    }
