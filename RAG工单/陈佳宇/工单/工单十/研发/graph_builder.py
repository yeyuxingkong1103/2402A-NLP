# graph_builder.py 知识图谱构建模块
# 工单编号：人工智能NLP‑RAG 项目‑金融问答系统部署
import os
import networkx as nx
from pyvis.network import Network


class GraphBuilder:
    def __init__(self):
        # 初始化有向图
        self.graph = nx.DiGraph()

    def add_triple(self, head, relation, tail):
        """添加三元组：头实体，关系，尾实体"""
        self.graph.add_node(head)
        self.graph.add_node(tail)
        self.graph.add_edge(head, tail, label=relation)

    def build_visual_html(self, output_path):
        """生成pyvis交互式图谱HTML文件"""
        net = Network(directed=True, height="700px", width="100%", bgcolor="#ffffff", font_color="#000000")
        # 加载节点
        for node in self.graph.nodes:
            net.add_node(node)
        # 加载边与关系标签
        for h, t, data in self.graph.edges(data=True):
            label = data.get("label", "")
            net.add_edge(h, t, label=label)
        net.write_html(output_path)
        return output_path

    def get_graph_info(self):
        """获取图谱统计信息：节点数量、边数量"""
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        return {"node": node_count, "edge": edge_count}


if __name__ == "__main__":
    # 本地简单测试
    gb = GraphBuilder()
    gb.add_triple("销售部", "包含", "大客户销售部")
    os.makedirs("data_output", exist_ok=True)
    out = gb.build_visual_html("data_output/graph_visual.html")
    print(f"测试可视化输出:{out}")
    print(gb.get_graph_info())
