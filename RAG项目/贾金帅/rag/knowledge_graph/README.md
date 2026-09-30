# 疾病-治疗知识图谱

## 数据文件

当前图谱使用项目根目录中的：

- 原始文件：疾病_药品_全量_长格式.json
- 英文字段文件：disease_drug_full_long_en.json

英文 schema：

| English key | 原字段 | 含义 |
|---|---|---|
| disease | 疾病 | 主疾病 |
| entity_name | 药品或物质 | 药品、食药物质、治疗措施或相关状况 |
| entity_type | 药型 | 西药、中药、非药物、非药品 |
| category | 类别 | 治疗药品、药物类别、食药物质、治疗措施、其他/备注 |
| applicable_scope | 适用证型/范围 | 证型、疾病分期、药物类别或适用场景 |
| description | 说明 | 关联说明 |
| usage | 用法 | 当前疾病和适用范围下的用法 |
| contraindications | 禁忌 | 当前疾病和适用范围下的禁忌 |
| evidence_source | 用法禁忌来源 | 指南、标准、药典或说明书来源 |

当前数据共有 337 条关联记录、12 种主疾病和 190 个不同名称。

## 为什么使用 Association 节点

同一种药品或食药物质可能出现在多个疾病、多个证型中，并且每条记录的用法、
禁忌和来源不同。如果把这些长文本直接写到 Treatment 节点上，后导入的疾病会
覆盖前一疾病的信息。

因此每一行数据建立一个 Association 节点，疾病特异信息保存在该节点上：

    (Association)-[:FOR_DISEASE]->(Disease)
    (Association)-[:USES_TREATMENT]->(Treatment)
    (Association)-[:MENTIONS_CONDITION]->(RelatedCondition)
    (Association)-[:APPLIES_TO]->(ClinicalScope)
    (Association)-[:SUPPORTED_BY]->(EvidenceSource)
    (Treatment)-[:IN_CATEGORY]->(TreatmentCategory)

一条 Association 只会连接 USES_TREATMENT 或 MENTIONS_CONDITION 中的一种。

## 实体

| Label | 含义 |
|---|---|
| Association | 一条疾病-实体关联，保存 description、usage、contraindications 等 |
| Disease | 12 种主疾病 |
| Treatment | 所有真实治疗实体的共同标签 |
| WesternDrug | entity_type=西药，同时具有 Treatment 标签 |
| HerbalSubstance | entity_type=中药，同时具有 Treatment 标签 |
| NonDrugIntervention | entity_type=非药物，同时具有 Treatment 标签 |
| RelatedCondition | entity_type=非药品的相关疾病、并发症、合并症或鉴别状况 |
| ClinicalScope | 证型、疾病分期、药物类别或适用场景 |
| EvidenceSource | 指南、标准、药典或说明书 |
| TreatmentCategory | 治疗药品、食药物质、治疗措施等类别 |

## 关系

| Relationship | 起点 | 终点 | 含义 |
|---|---|---|---|
| FOR_DISEASE | Association | Disease | 关联记录属于哪个疾病 |
| USES_TREATMENT | Association | Treatment | 关联记录采用哪个治疗实体 |
| MENTIONS_CONDITION | Association | RelatedCondition | 记录中提到的相关临床状况 |
| APPLIES_TO | Association | ClinicalScope | 适用证型、分期或范围 |
| SUPPORTED_BY | Association | EvidenceSource | 证据来源 |
| IN_CATEGORY | Treatment | TreatmentCategory | 治疗实体所属类别 |

## 导入

只校验数据，不连接 Neo4j：

    python -m src.knowledge_graph.data_to_neo4j --dry-run

首次导入新模型、并明确替换旧图谱：

    python -m src.knowledge_graph.data_to_neo4j --replace

后续增量或幂等更新：

    python -m src.knowledge_graph.data_to_neo4j

--replace 会删除旧图谱和新模型的受管节点，请只在确认需要替换时使用。导入脚本使用
MERGE 和稳定 association_id，同一文件重复导入不会制造重复节点。

## 查询

命令行自然语言查询：

    python -m src.knowledge_graph.graph_retrieval "高脂血症有哪些治疗药品？"
    python -m src.knowledge_graph.graph_retrieval "阴虚证有哪些食药物质？" --show-context
    python -m src.knowledge_graph.graph_retrieval "阿托伐他汀有哪些禁忌？"

查询某疾病的治疗实体：

    MATCH (a:Association)-[:FOR_DISEASE]->(d:Disease {name: '高脂血症'})
    MATCH (a)-[:USES_TREATMENT]->(t:Treatment)
    OPTIONAL MATCH (a)-[:APPLIES_TO]->(s:ClinicalScope)
    RETURN t.name, t.normalized_type, a.usage, a.contraindications,
           collect(DISTINCT s.name) AS scopes;

查询某个证型对应的食药物质：

    MATCH (a:Association)-[:APPLIES_TO]->(s:ClinicalScope {name: '阴虚证'})
    MATCH (a)-[:USES_TREATMENT]->(t:HerbalSubstance)
    MATCH (a)-[:FOR_DISEASE]->(d:Disease)
    RETURN d.name, t.name, a.usage, a.contraindications;

查询一个治疗实体在不同疾病下的用法与禁忌：

    MATCH (a:Association)-[:USES_TREATMENT]->(t:Treatment {name: '黄芪'})
    MATCH (a)-[:FOR_DISEASE]->(d:Disease)
    OPTIONAL MATCH (a)-[:APPLIES_TO]->(s:ClinicalScope)
    RETURN d.name, s.name, a.usage, a.contraindications, a.evidence_source;

查询疾病相关状况：

    MATCH (a:Association)-[:FOR_DISEASE]->(d:Disease {name: '痛风'})
    MATCH (a)-[:MENTIONS_CONDITION]->(c:RelatedCondition)
    RETURN c.name, a.description, a.evidence_source;

查询证据来源：

    MATCH (a:Association)-[:SUPPORTED_BY]->(source:EvidenceSource)
    MATCH (a)-[:FOR_DISEASE]->(d:Disease)
    OPTIONAL MATCH (a)-[:USES_TREATMENT]->(t:Treatment)
    WHERE source.name CONTAINS '中国2型糖尿病防治指南'
    RETURN source.name, d.name, t.name;

## 项目兼容性

graph_retrieval.py 保留了 DrugGraphRetriever 和 HybridGraphRetriever 类名，因此
app.py 与统一混合检索路由不需要修改。新代码同时提供更准确的别名
ClinicalGraphRetriever。
