# -*- coding: utf-8 -*-
"""预置的参数化 Cypher 模板（静态常量，不含任何执行逻辑）。

自 ``graph_retrieval.py`` 拆出。
安全约定：LLM **不生成** Cypher，只输出结构化 plan；这里按 plan 选模板、
用参数绑定执行，从根本上规避 Cypher 注入与幻觉查询。
"""

# Static parameterized Cypher only; the LLM never generates executable Cypher.
RETRIEVAL_QUERY = """
MATCH (association:Association)-[:FOR_DISEASE]->(disease:Disease)
OPTIONAL MATCH (association)-[:USES_TREATMENT]->(treatment:Treatment)
OPTIONAL MATCH (association)-[:MENTIONS_CONDITION]->(
    relatedCondition:RelatedCondition
)
WHERE any(keyword IN $keywords WHERE
    toLower(disease.name) CONTAINS toLower(keyword)
    OR toLower(coalesce(treatment.name, '')) CONTAINS toLower(keyword)
    OR toLower(coalesce(relatedCondition.name, '')) CONTAINS toLower(keyword)
    OR toLower(coalesce(association.applicable_scope, '')) CONTAINS toLower(keyword)
    OR toLower(coalesce(association.category, '')) CONTAINS toLower(keyword)
    OR toLower(coalesce(association.entity_type, '')) CONTAINS toLower(keyword)
    OR toLower(coalesce(association.evidence_source, '')) CONTAINS toLower(keyword)
    OR EXISTS {
        MATCH (association)-[:APPLIES_TO]->(scope:ClinicalScope)
        WHERE toLower(scope.name) CONTAINS toLower(keyword)
    }
    OR EXISTS {
        MATCH (association)-[:SUPPORTED_BY]->(source:EvidenceSource)
        WHERE toLower(source.name) CONTAINS toLower(keyword)
    }
    OR EXISTS {
        MATCH (treatment)-[:IN_CATEGORY]->(category:TreatmentCategory)
        WHERE toLower(category.name) CONTAINS toLower(keyword)
    }
)
WITH association, disease, treatment, relatedCondition
ORDER BY disease.name, coalesce(treatment.name, relatedCondition.name),
         association.applicable_scope
LIMIT $limit
OPTIONAL MATCH (association)-[:APPLIES_TO]->(scope:ClinicalScope)
OPTIONAL MATCH (association)-[:SUPPORTED_BY]->(source:EvidenceSource)
OPTIONAL MATCH (treatment)-[:IN_CATEGORY]->(category:TreatmentCategory)
RETURN association.id AS id,
       disease.name AS disease,
       coalesce(treatment.name, relatedCondition.name) AS entity_name,
       CASE WHEN treatment IS NOT NULL
            THEN 'treatment' ELSE 'related_condition' END AS target_kind,
       association.entity_type AS entity_type,
       association.normalized_type AS normalized_type,
       association.category AS category,
       association.applicable_scope AS applicable_scope,
       association.description AS description,
       association.usage AS usage,
       association.contraindications AS contraindications,
       association.evidence_source AS evidence_source,
       collect(DISTINCT scope.name) AS scopes,
       collect(DISTINCT source.name) AS evidence_sources,
       collect(DISTINCT category.name) AS categories
ORDER BY disease, entity_name, applicable_scope
"""


DISEASE_TREATMENT_QUERY = """
MATCH (association:Association)-[:FOR_DISEASE]->(disease:Disease)
MATCH (association)-[:USES_TREATMENT]->(treatment:Treatment)
WHERE disease.name = $disease
   OR toLower(disease.name) CONTAINS toLower($disease)
   OR toLower($disease) CONTAINS toLower(disease.name)
OPTIONAL MATCH (association)-[:APPLIES_TO]->(scope:ClinicalScope)
OPTIONAL MATCH (association)-[:SUPPORTED_BY]->(source:EvidenceSource)
OPTIONAL MATCH (treatment)-[:IN_CATEGORY]->(category:TreatmentCategory)
WITH association, disease, treatment,
     collect(DISTINCT scope.name) AS scopes,
     collect(DISTINCT source.name) AS evidence_sources,
     collect(DISTINCT category.name) AS categories
RETURN association.id AS id,
       disease.name AS disease,
       treatment.name AS entity_name,
       'treatment' AS target_kind,
       association.entity_type AS entity_type,
       association.normalized_type AS normalized_type,
       association.category AS category,
       association.applicable_scope AS applicable_scope,
       association.description AS description,
       association.usage AS usage,
       association.contraindications AS contraindications,
       association.evidence_source AS evidence_source,
       scopes, evidence_sources, categories
ORDER BY CASE association.normalized_type
             WHEN 'western_drug' THEN 0
             WHEN 'non_drug_intervention' THEN 1
             ELSE 2
         END,
         entity_name, applicable_scope
LIMIT $limit
"""


EVIDENCE_DISEASE_TREATMENT_QUERY = """
MATCH (association:Association)-[:FOR_DISEASE]->(disease:Disease)
MATCH (association)-[:USES_TREATMENT]->(treatment:Treatment)
UNWIND range(0, size($evidence_texts) - 1) AS evidence_rank
WITH association, disease, treatment, evidence_rank,
     $evidence_texts[evidence_rank] AS evidence,
     trim(head(split(disease.name, '（'))) AS base_disease_name
WHERE toLower(evidence) CONTAINS toLower(disease.name)
   OR (
       size(base_disease_name) >= 2
       AND toLower(evidence) CONTAINS toLower(base_disease_name)
   )
WITH association, disease, treatment, min(evidence_rank) AS vector_rank
OPTIONAL MATCH (association)-[:APPLIES_TO]->(scope:ClinicalScope)
OPTIONAL MATCH (association)-[:SUPPORTED_BY]->(source:EvidenceSource)
OPTIONAL MATCH (treatment)-[:IN_CATEGORY]->(category:TreatmentCategory)
WITH association, disease, treatment, vector_rank,
     collect(DISTINCT scope.name) AS scopes,
     collect(DISTINCT source.name) AS evidence_sources,
     collect(DISTINCT category.name) AS categories
RETURN association.id AS id,
       disease.name AS disease,
       treatment.name AS entity_name,
       'treatment' AS target_kind,
       association.entity_type AS entity_type,
       association.normalized_type AS normalized_type,
       association.category AS category,
       association.applicable_scope AS applicable_scope,
       association.description AS description,
       association.usage AS usage,
       association.contraindications AS contraindications,
       association.evidence_source AS evidence_source,
       vector_rank,
       scopes, evidence_sources, categories
ORDER BY vector_rank, disease, entity_name, applicable_scope
LIMIT $limit
"""
