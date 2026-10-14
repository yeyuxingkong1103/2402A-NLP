# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""组织隶属（BELONGS_TO）抽取 + 组织树检索的规格测试。

背景：Q10/Q13 是 eval_question 里的组织结构题，而图谱此前没有任何
BELONGS_TO 边（`find_sectors` 只造 Department 节点、不造父子边），
导致 org_tree 检索模式空转。本文件锁定「文本 → 组织关系 → 组织树检索」
这条链路的规格。

文本样本全部取自本工程真实语料（平安银行 / 中信证券年报）。
"""
from src.graph_retriever import retrieve_subgraph
from src.kg_dict import COMPANY_TABLE
from src.kg_extract import extract_block, find_org_relations
from src.kg_schema import Entity, Relation
from src.kg_store import KnowledgeGraph

# 真实语料（平安银行 2019 年报·董事会下设专门委员会）
TEXT_DENG = ("8.7 董事会下设专门委员会在报告期内履行职责情况 本行第十届董事会"
             "和第十一届董事会均下设战略发展与消费者权益保护委员会、审计委员会、"
             "风险管理委员会、关联交易控制委员会、提名委员会和薪酬与考核委员会"
             "等6 个专门委员会。")
# 真实语料（中信证券 2020 年报·附录一组织结构图 注1）
TEXT_CITIC = ("注1：投资银行管理委员会下设金融行业组、能源化工行业组、"
              "基础设施与房地产行业组、装备制造行业组、信息传媒行业组、"
              "医疗健康行业组、消费行业组、综合行业组（北京）、综合行业组（上海）。")


def _linked(name, etype="Company"):
    return [{"entity_id": f"{etype}:{name}", "name": name, "type": etype,
             "score": 0.9, "surface": name}]


# ---------------------------------------------------------------------------
# 一、文本 → 组织关系（find_org_relations）
# ---------------------------------------------------------------------------
def test_find_org_relations_basic():
    groups = find_org_relations(TEXT_DENG)
    assert len(groups) == 1
    g = groups[0]
    assert g["parent"] == "董事会"          # 剥掉「本行第十届」等修饰
    assert g["parent_type"] == "Department"
    assert g["children"] == ["战略发展与消费者权益保护委员会", "审计委员会",
                             "风险管理委员会", "关联交易控制委员会",
                             "提名委员会", "薪酬与考核委员会"]   # 「和」也要切开
    assert g["trigger"] == "下设"


def test_find_org_relations_keeps_parenthesized_children():
    """「综合行业组（北京）」这类带括号的子节点名字必须完整保留。"""
    groups = find_org_relations(TEXT_CITIC)
    assert groups[0]["parent"] == "投资银行管理委员会"
    kids = groups[0]["children"]
    assert "综合行业组（北京）" in kids
    assert "金融行业组" in kids
    assert len(kids) == 9


def test_find_org_relations_evidence_is_原文片段():
    g = find_org_relations(TEXT_DENG)[0]
    assert "董事会" in g["evidence"] and "下设" in g["evidence"]
    assert g["evidence"] in TEXT_DENG


def test_find_org_relations_rejects_non_org_head():
    """「高级管理层及其下设委员会负责…」的头不是组织单元，不得建边。"""
    assert find_org_relations(
        "高级管理层及其下设委员会负责在董事会授权范围内审批重大事项。") == []


def test_find_org_relations_rejects_generic_children():
    """「总行内设部门及直属单位分为…」是泛指，不产物名。"""
    assert find_org_relations(
        "本行总行内设部门及直属单位分为个人金融板块和公司金融板块。") == []


def test_find_org_relations_needs_two_children():
    """只有一个子项的句子不算组织树（多为噪声）。"""
    assert find_org_relations("董事会下设审计委员会负责监督。") == []


def test_find_org_relations_parent_may_be_company():
    """头实体是已知公司时，父节点类型应记为 Company。"""
    g = find_org_relations("中信证券下设财富管理委员会、投资银行委员会。")[0]
    assert g["parent"] == "中信证券"
    assert g["parent_type"] == "Company"
    assert g["parent"] in COMPANY_TABLE


# 真实语料（中国平安 2019 年报·双栏 PDF 解析后行交错，物名被拦腰打断）
TEXT_INTERLEAVED = (
    "董事会下设审计与风险管理委员会，全面了\n委员会、投资者关系及ESG委员会、"
    "全球系统重要性保险机\n解公司面临的各项重大风险及其管理状况，监督风险管理体\n"
    "构管理委员会、科技发展委员会等8个管理委员会。")


def test_find_org_relations_rejects_interleaved_garbled_list():
    """「等8个」只对出 4 个完整名字 → 双栏文本已损坏，整组不抽（宁可漏抽）。"""
    assert find_org_relations(TEXT_INTERLEAVED) == []


def test_find_org_relations_keeps_list_matching_等N个():
    """名单与「等N个」数量一致时正常抽取。"""
    g = find_org_relations(
        "本行董事会下设审计委员会、风险管理委员会等2个专门委员会。")
    assert g[0]["children"] == ["审计委员会", "风险管理委员会"]


def test_find_org_relations_rejects_sentence_head():
    """「各位监事分别列席董事会下设…」的头是句子片段，不是组织名。"""
    assert find_org_relations(
        "报告期内，各位监事分别列席董事会下设审计委员会、提名委员会。") == []


# ---------------------------------------------------------------------------
# 二、块级抽取：产出实体 + BELONGS_TO 边（含 provenance）
# ---------------------------------------------------------------------------
def test_extract_block_emits_belongs_to_child_to_parent():
    chunk = {"text": TEXT_DENG, "source": "2020-02-14__平安银行__000001__2019年度报告.pdf",
             "page_idx": 39}
    out = extract_block(chunk)
    ids = {e.entity_id for e in out["entities"]}
    assert "Department:董事会" in ids
    assert "Department:审计委员会" in ids

    edges = [r for r in out["relations"] if r.rel_type == "BELONGS_TO"]
    # 本体规定 BELONGS_TO 为「子 → 父」（部门指向所属组织）
    assert ("Department:审计委员会", "Department:董事会") in {
        (r.head_id, r.tail_id) for r in edges}
    assert len(edges) == 6


def test_extract_block_belongs_to_carries_provenance():
    chunk = {"text": TEXT_DENG, "source": "2020-02-14__平安银行__000001__2019年度报告.pdf",
             "page_idx": 39}
    edge = [r for r in extract_block(chunk)["relations"]
            if r.rel_type == "BELONGS_TO"][0]
    assert edge.source_doc.endswith("2019年度报告.pdf")
    assert edge.page == 39
    assert "下设" in edge.evidence
    assert edge.confidence == 1.0          # 规则抽取，非 LLM，置信度满


# ---------------------------------------------------------------------------
# 三、组织树检索：父子方向不得颠倒
# ---------------------------------------------------------------------------
def _org_kg():
    kg = KnowledgeGraph()
    kg.add_entity(Entity("Company:平安银行", "平安银行", "Company"))
    kg.add_entity(Entity("Department:董事会", "董事会", "Department"))
    kg.add_entity(Entity("Department:审计委员会", "审计委员会", "Department"))
    kg.add_relation(Relation("Department:董事会", "BELONGS_TO", "Company:平安银行",
                             evidence="董事会", source_doc="a.pdf", page=5))
    kg.add_relation(Relation("Department:审计委员会", "BELONGS_TO", "Department:董事会",
                             evidence="董事会下设审计委员会", source_doc="a.pdf", page=5))
    return kg


def test_org_tree_facts_are_parent_to_children():
    kg = _org_kg()
    sub = retrieve_subgraph("平安银行的组织结构图中董事会下设哪些委员会？",
                            _linked("平安银行"), kg)
    assert sub.mode == "org_tree"
    joined = "\n".join(sub.facts)
    assert "董事会 下设：审计委员会" in joined, sub.facts
    assert "平安银行 下设：董事会" in joined, sub.facts
    # 方向反了会把子节点当成父节点陈述
    assert "审计委员会 下设" not in joined, sub.facts


def test_org_tree_facts_include_graph_nodes_for_visualization():
    kg = _org_kg()
    sub = retrieve_subgraph("平安银行的组织结构图中董事会下设哪些委员会？",
                            _linked("平安银行"), kg)
    node_ids = {n["id"] for n in sub.nodes}
    assert {"Company:平安银行", "Department:董事会", "Department:审计委员会"} <= node_ids
    assert any(e["label"] == "BELONGS_TO" for e in sub.edges)
