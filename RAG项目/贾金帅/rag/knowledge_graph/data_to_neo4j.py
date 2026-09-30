"""Import the English disease-treatment dataset into Neo4j.

Graph model (one Association per source row):

    (Association)-[:FOR_DISEASE]->(Disease)
    (Association)-[:USES_TREATMENT]->(Treatment)
    (Association)-[:MENTIONS_CONDITION]->(RelatedCondition)
    (Association)-[:APPLIES_TO]->(ClinicalScope)
    (Association)-[:SUPPORTED_BY]->(EvidenceSource)
    (Treatment)-[:IN_CATEGORY]->(TreatmentCategory)

Only one of USES_TREATMENT / MENTIONS_CONDITION is created for each row.
Disease-specific usage and contraindication text is stored on Association so
the same treatment can safely have different guidance for different diseases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "disease_drug_full_long_en.json"
DEFAULT_URI = "bolt://127.0.0.1:7687"
DEFAULT_USERNAME = "neo4j"
DEFAULT_PASSWORD = "medical-rag-2026"
SOURCE_FILE = DEFAULT_INPUT.name

REQUIRED_FIELDS = (
    "disease",
    "entity_name",
    "entity_type",
    "category",
    "applicable_scope",
    "description",
    "usage",
    "contraindications",
    "evidence_source",
)

ENTITY_TYPE_MAP = {
    "西药": "western_drug",
    "中药": "herbal_substance",
    "非药物": "non_drug_intervention",
    "非药品": "related_condition",
}

MANAGED_RELATIONSHIPS = (
    "FOR_DISEASE|USES_TREATMENT|MENTIONS_CONDITION|APPLIES_TO|SUPPORTED_BY"
)

CONSTRAINTS = [
    "CREATE CONSTRAINT association_id IF NOT EXISTS "
    "FOR (n:Association) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT disease_name IF NOT EXISTS "
    "FOR (n:Disease) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT treatment_name IF NOT EXISTS "
    "FOR (n:Treatment) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT related_condition_name IF NOT EXISTS "
    "FOR (n:RelatedCondition) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT clinical_scope_name IF NOT EXISTS "
    "FOR (n:ClinicalScope) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT evidence_source_name IF NOT EXISTS "
    "FOR (n:EvidenceSource) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT treatment_category_name IF NOT EXISTS "
    "FOR (n:TreatmentCategory) REQUIRE n.name IS UNIQUE",
]

# --replace is deliberately explicit because this removes the legacy graph too.
CLEAR_MANAGED_GRAPH_QUERY = """
MATCH (node)
WHERE any(label IN labels(node) WHERE label IN [
    'Association', 'Treatment', 'WesternDrug', 'HerbalSubstance',
    'NonDrugIntervention', 'RelatedCondition', 'ClinicalScope',
    'EvidenceSource', 'TreatmentCategory',
    'DrugProduct', 'Drug', 'Ingredient', 'Manufacturer', 'SubstanceSource',
    'RegulatoryClass', 'PharmacologicalClass', 'DosageForm', 'InsuranceClass'
])
DETACH DELETE node
"""

DELETE_ORPHAN_DISEASES_QUERY = """
MATCH (disease:Disease)
WHERE NOT (disease)--()
DELETE disease
"""

IMPORT_QUERY = f"""
UNWIND $rows AS row
MERGE (association:Association {{id: row.association_id}})
SET association += row.properties
WITH row, association
OPTIONAL MATCH (association)-[old:{MANAGED_RELATIONSHIPS}]->()
WITH row, association, collect(old) AS old_relationships
FOREACH (relationship IN old_relationships | DELETE relationship)

MERGE (disease:Disease {{name: row.disease}})
MERGE (association)-[:FOR_DISEASE]->(disease)

FOREACH (_ IN CASE WHEN row.target_kind = 'treatment' THEN [1] ELSE [] END |
    MERGE (treatment:Treatment {{name: row.entity_name}})
    SET treatment.entity_type = row.entity_type,
        treatment.normalized_type = row.normalized_type,
        treatment.category = row.category,
        treatment.source_file = row.source_file
    MERGE (association)-[:USES_TREATMENT]->(treatment)
)

FOREACH (_ IN CASE WHEN row.normalized_type = 'western_drug' THEN [1] ELSE [] END |
    MERGE (treatment:Treatment {{name: row.entity_name}})
    SET treatment:WesternDrug
)
FOREACH (_ IN CASE WHEN row.normalized_type = 'herbal_substance' THEN [1] ELSE [] END |
    MERGE (treatment:Treatment {{name: row.entity_name}})
    SET treatment:HerbalSubstance
)
FOREACH (_ IN CASE WHEN row.normalized_type = 'non_drug_intervention' THEN [1] ELSE [] END |
    MERGE (treatment:Treatment {{name: row.entity_name}})
    SET treatment:NonDrugIntervention
)

FOREACH (_ IN CASE WHEN row.target_kind = 'related_condition' THEN [1] ELSE [] END |
    MERGE (condition:RelatedCondition {{name: row.entity_name}})
    SET condition.source_file = row.source_file
    MERGE (association)-[:MENTIONS_CONDITION]->(condition)
)

FOREACH (_ IN CASE
    WHEN row.target_kind = 'treatment' AND row.category <> '' THEN [1] ELSE [] END |
    MERGE (treatment:Treatment {{name: row.entity_name}})
    MERGE (category:TreatmentCategory {{name: row.category}})
    MERGE (treatment)-[:IN_CATEGORY]->(category)
)

FOREACH (_ IN CASE WHEN row.applicable_scope <> '' THEN [1] ELSE [] END |
    MERGE (scope:ClinicalScope {{name: row.applicable_scope}})
    MERGE (association)-[:APPLIES_TO]->(scope)
)

FOREACH (source_name IN row.evidence_sources |
    MERGE (source:EvidenceSource {{name: source_name}})
    MERGE (association)-[:SUPPORTED_BY]->(source)
)
"""


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def split_sources(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    return list(dict.fromkeys(
        part.strip() for part in re.split(r"[；;]", text) if part.strip()
    ))


def association_id(record: dict[str, Any]) -> str:
    identity = json.dumps(
        [clean_text(record.get(field)) for field in REQUIRED_FIELDS],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def prepare_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    missing_keys = [field for field in REQUIRED_FIELDS if field not in record]
    if missing_keys:
        raise ValueError(
            f"Record {index + 1} is missing fields: {', '.join(missing_keys)}"
        )

    disease = clean_text(record["disease"])
    entity_name = clean_text(record["entity_name"])
    entity_type = clean_text(record["entity_type"])
    if not disease or not entity_name:
        raise ValueError(f"Record {index + 1} requires disease and entity_name")
    if entity_type not in ENTITY_TYPE_MAP:
        raise ValueError(
            f"Record {index + 1} has unsupported entity_type: {entity_type!r}"
        )

    normalized_type = ENTITY_TYPE_MAP[entity_type]
    target_kind = (
        "related_condition"
        if normalized_type == "related_condition"
        else "treatment"
    )
    category = clean_text(record["category"])
    scope = clean_text(record["applicable_scope"])
    source_text = clean_text(record["evidence_source"])
    properties = {
        "disease_name": disease,
        "entity_name": entity_name,
        "entity_type": entity_type,
        "normalized_type": normalized_type,
        "category": category,
        "applicable_scope": scope,
        "description": clean_text(record["description"]),
        "usage": clean_text(record["usage"]),
        "contraindications": clean_text(record["contraindications"]),
        "evidence_source": source_text,
        "source_file": SOURCE_FILE,
    }

    return {
        "association_id": association_id(record),
        "properties": properties,
        "disease": disease,
        "entity_name": entity_name,
        "entity_type": entity_type,
        "normalized_type": normalized_type,
        "target_kind": target_kind,
        "category": category,
        "applicable_scope": scope,
        "evidence_sources": split_sources(source_text),
        "source_file": SOURCE_FILE,
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError("JSON root must be an array")

    rows_by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index + 1} is not an object")
        row = prepare_record(record, index)
        row["source_file"] = path.name
        row["properties"]["source_file"] = path.name
        rows_by_id[row["association_id"]] = row
    return list(rows_by_id.values())


def batches(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def graph_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "associations": len(rows),
        "diseases": len({row["disease"] for row in rows}),
        "treatments": len({
            row["entity_name"] for row in rows
            if row["target_kind"] == "treatment"
        }),
        "related_conditions": len({
            row["entity_name"] for row in rows
            if row["target_kind"] == "related_condition"
        }),
        "clinical_scopes": len({
            row["applicable_scope"] for row in rows if row["applicable_scope"]
        }),
        "evidence_sources": len({
            source for row in rows for source in row["evidence_sources"]
        }),
        "categories": len({
            row["category"] for row in rows
            if row["target_kind"] == "treatment" and row["category"]
        }),
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    print(json.dumps(graph_summary(rows), ensure_ascii=False, indent=2))


def import_rows(
    rows: list[dict[str, Any]],
    uri: str,
    username: str,
    password: str,
    database: str,
    batch_size: int,
    replace: bool = False,
) -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError as error:
        raise RuntimeError("Missing dependency: pip install neo4j") from error

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            if replace:
                session.run(CLEAR_MANAGED_GRAPH_QUERY).consume()
                session.run(DELETE_ORPHAN_DISEASES_QUERY).consume()
                print("Legacy and current managed graph data removed.")
            for query in CONSTRAINTS:
                session.run(query).consume()
            for number, batch in enumerate(batches(rows, batch_size), start=1):
                session.execute_write(
                    lambda transaction, current=batch: transaction.run(
                        IMPORT_QUERY, rows=current
                    ).consume()
                )
                print(f"Imported batch {number}: {len(batch)} rows")
    finally:
        driver.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import disease-treatment English JSON into Neo4j"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", DEFAULT_URI))
    parser.add_argument("--username", default=os.getenv("NEO4J_USER", DEFAULT_USERNAME))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", DEFAULT_PASSWORD))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete legacy/current managed graph nodes before importing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize JSON without connecting to Neo4j",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be greater than zero")
    rows = load_rows(args.input)
    print_summary(rows)
    if args.dry_run:
        print("Validation passed; Neo4j was not modified.")
        return
    import_rows(
        rows,
        args.uri,
        args.username,
        args.password,
        args.database,
        args.batch_size,
        replace=args.replace,
    )
    print("Neo4j import completed.")


if __name__ == "__main__":
    main()
