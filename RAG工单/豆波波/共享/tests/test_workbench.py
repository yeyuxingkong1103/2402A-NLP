import unittest

from rag_workbench.core import build_graph, chunk_text, graph_search, hybrid_search, lexical_search


class WorkbenchTests(unittest.TestCase):
    def test_chunk_and_search(self):
        docs = chunk_text("工业设备包括泵和阀门。泵负责输送流体。", "demo", size=12, overlap=2)
        self.assertTrue(docs)
        self.assertGreater(lexical_search("泵", docs)[0]["score"], 0)
        self.assertGreater(hybrid_search("阀门", docs)[0]["score"], 0)
        self.assertTrue(graph_search("泵", build_graph(docs), docs))
