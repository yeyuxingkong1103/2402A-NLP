import tempfile
import unittest
from pathlib import Path

from corpus import Chunk
from graph_rag import GraphRAG, build_graph, save_index


def sample_chunks():
    return [
        Chunk("000001-p5-1", "平安银行2019年年报", "000001", "平安银行", 2019, 5, "营业收入1,379.58亿元，净利润281.95亿元。", "主要会计数据"),
        Chunk("601318-p10-1", "中国平安2019年年报", "601318", "中国平安", 2019, 10, "营业收入1,168,867百万元，净利润149,407百万元。", "经营业绩"),
        Chunk("600030-p17-1", "中信证券2020年年报", "600030", "中信证券", 2020, 17, "营业收入543.83亿元。", "财务摘要"),
    ]


class KnowledgeGraphTests(unittest.TestCase):
    def test_graph_contains_company_report_year_metric_and_chunk_relations(self):
        graph = build_graph(sample_chunks())
        types = {node["type"] for node in graph["nodes"]}
        relations = {edge["relation"] for edge in graph["edges"]}
        self.assertEqual(types, {"company", "report", "year", "metric", "chunk"})
        self.assertTrue({"HAS_REPORT", "REPORT_YEAR", "HAS_CHUNK", "MENTIONS"}.issubset(relations))
        self.assertEqual(len([node for node in graph["nodes"] if node["type"] == "chunk"]), 3)

    def test_graph_search_promotes_named_company_and_returns_related_subgraph(self):
        rag = GraphRAG.fit(sample_chunks())
        result = rag.search("平安银行2019年的营业收入和净利润是多少？", top_k=2)
        self.assertEqual(result["hits"][0]["stock_code"], "000001")
        self.assertEqual(result["intent"]["stocks"], ["000001"])
        node_types = {node["type"] for node in result["graph"]["nodes"]}
        self.assertTrue({"company", "report", "year", "metric", "chunk"}.issubset(node_types))
        self.assertLessEqual(len([node for node in result["graph"]["nodes"] if node["type"] == "chunk"]), 2)

    def test_multi_company_query_keeps_one_seed_from_each_named_company(self):
        rag = GraphRAG.fit(sample_chunks())
        result = rag.search("比较平安银行、中国平安和中信证券的营业收入", top_k=3)
        self.assertEqual({hit["stock_code"] for hit in result["hits"]}, {"000001", "601318", "600030"})

    def test_subgraph_does_not_expand_from_report_to_unselected_chunks(self):
        chunks = [
            Chunk(f"000001-p{page}-1", "平安银行2019年年报", "000001", "平安银行", 2019, page, f"第{page}页营业收入说明", "财务数据")
            for page in range(1, 31)
        ]
        result = GraphRAG.fit(chunks).search("平安银行营业收入", top_k=1)
        chunk_nodes = [node for node in result["graph"]["nodes"] if node["type"] == "chunk"]
        self.assertEqual(len(chunk_nodes), 1)
        self.assertLessEqual(len(result["graph"]["nodes"]), 6)

    def test_group_level_financial_summary_beats_subsidiary_metric(self):
        chunks = [
            Chunk("601211-p38-1", "国泰君安2021年年报", "601211", "国泰君安", 2021, 38, "国泰君安期货实现营业收入112.98亿元。", "子公司情况"),
            Chunk("601211-p29-1", "国泰君安2021年年报", "601211", "国泰君安", 2021, 29, "2021年，本集团实现营业收入428.17亿元，净利润150.13亿元。", "主要会计数据"),
        ]
        result = GraphRAG.fit(chunks).search("国泰君安2021年的营业收入是多少？", top_k=1)
        self.assertEqual(result["hits"][0]["chunk_id"], "601211-p29-1")

    def test_empty_query_and_top_k_are_validated(self):
        rag = GraphRAG.fit(sample_chunks())
        with self.assertRaisesRegex(ValueError, "问题不能为空"):
            rag.search("   ")
        with self.assertRaisesRegex(ValueError, "Top K"):
            rag.search("营业收入", top_k=0)

    def test_saved_index_round_trips_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            save_index(sample_chunks(), data_dir)
            loaded = GraphRAG.load(data_dir)
            result = loaded.search("中信证券2020年营业收入", top_k=1)
        self.assertEqual(result["hits"][0]["chunk_id"], "600030-p17-1")


if __name__ == "__main__":
    unittest.main()

