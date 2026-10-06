# -*- coding: utf-8 -*-
# graph_builder.py 工单8 GraphRAG 图谱构建模块
# 工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
import networkx as nx
from pyvis.network import Network

class GraphBuilder:
    def __init__(self):
        # 创建内存知识图谱
        self.graph = nx.DiGraph()

    def add_triple(self, head_entity: str, relation: str, tail_entity: str):
        """添加三元组 (头实体，关系，尾实体)"""
        self.graph.add_node(head_entity, label=head_entity)
        self.graph.add_node(tail_entity, label=tail_entity)
        self.graph.add_edge(head_entity, tail_entity, label=relation)

    def build_visual_html(self, output_file="graph_visual.html"):
        """生成知识图谱可视化网页，直接打开即可查看图谱"""
        net = Network(directed=True, height="700px", width="100%")
        net.from_nx(self.graph)
        net.write_html(output_file)
        print(f"✅知识图谱可视化已输出到：{output_file}")
        return output_file

    def get_graph_info(self):
        """获取图谱统计信息"""
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        return {"node": node_count, "edge": edge_count}


if __name__ == "__main__":
    # 测试样例三元组
    gb = GraphBuilder()
    gb.add_triple("武汉力源信息技术股份有限公司", "下设部门", "销售部")
    gb.add_triple("销售部", "包含", "大客户销售部")
    gb.add_triple("大客户销售部", "下辖", "华北销售处")
    gb.add_triple("大客户销售部", "下辖", "华东销售处")
    gb.build_visual_html("demo_graph.html")
    print("图谱统计：", gb.get_graph_info())
