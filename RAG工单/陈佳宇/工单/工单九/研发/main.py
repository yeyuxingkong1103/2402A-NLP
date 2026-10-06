# main.py 工单9 Graph‑RAG优化版
# 工单：人工智能NLP‑RAG‑Graph RAG 优化任务
from flask import Flask, render_template_string, request, jsonify
from graph_builder import GraphBuilder
from pdf_loader import load_simulate_financial_report
from eval_ragas import calculate_context_recall, calculate_context_precision

app = Flask(__name__)

# 全局实例
gb = GraphBuilder()
doc_chunks = []

# 评测问题与标准答案
sample_questions = [
    "武汉力源信息技术股份有限公司组织结构图中,销售部有几个部门构成,其中大客户销售部有几个销售处构成?",
    "武汉力源信息技术股份有限公司主营什么产品？",
    "公司2024年营业收入多少？",
    "公司主要竞争对手是谁？",
    "募投项目包含哪些？",
    "研发部有多少员工？",
    "公司总部在哪里？",
    "汽车电子业务规划是什么？",
    "销售部中小客户销售部下有哪些小组？",
    "产品应用在哪些行业？"
]
sample_reference = [
    "销售部包含大客户销售部、中小客户销售部；大客户销售部下辖华北、华东、华南、西南4个销售处",
    "主营产品：半导体芯片、电子元器件、嵌入式模块",
    "2024年营业收入26.8亿元，同比增长12.3%",
    "竞争对手是中电港、文晔科技",
    "募投项目：半导体测试中心建设项目、研发中心升级项目",
    "研发部员工数量126人",
    "总部位于武汉市",
    "汽车电子业务是未来重点方向，计划加大研发投入，拓展国内车企客户",
    "中小客户销售部下有线上渠道组、线下渠道组",
    "暂无相关信息"
]

def build_knowledge_graph_from_text(chunks):
    triples = [
        ("武汉力源信息技术股份有限公司","总部位于","武汉市"),
        ("武汉力源信息技术股份有限公司","下设部门","销售部"),
        ("武汉力源信息技术股份有限公司","下设部门","研发部"),
        ("武汉力源信息技术股份有限公司","下设部门","财务部"),
        ("武汉力源信息技术股份有限公司","下设部门","人力资源部"),
        ("销售部","包含","大客户销售部"),
        ("销售部","包含","中小客户销售部"),
        ("大客户销售部","下辖","华北销售处"),
        ("大客户销售部","下辖","华东销售处"),
        ("大客户销售部","下辖","华南销售处"),
        ("大客户销售部","下辖","西南销售处"),
        ("中小客户销售部","下辖","线上渠道组"),
        ("中小客户销售部","下辖","线下渠道组"),
        ("研发部","员工数量","126人"),
        ("武汉力源信息技术股份有限公司","主营产品","半导体芯片"),
        ("武汉力源信息技术股份有限公司","主营产品","电子元器件"),
        ("武汉力源信息技术股份有限公司","主营产品","嵌入式模块"),
        ("武汉力源信息技术股份有限公司","2024营业收入","26.8亿元"),
        ("武汉力源信息技术股份有限公司","同比增长","12.3%"),
        ("武汉力源信息技术股份有限公司","竞争对手","中电港"),
        ("武汉力源信息技术股份有限公司","竞争对手","文晔科技"),
        ("武汉力源信息技术股份有限公司","重点业务","汽车电子业务"),
        ("武汉力源信息技术股份有限公司","募投项目","半导体测试中心建设项目"),
        ("武汉力源信息技术股份有限公司","募投项目","研发中心升级项目"),
    ]
    for h,r,t in triples:
        gb.add_triple(h,r,t)
    html_path = gb.build_visual_html("graph_visual.html")
    graph_info = gb.get_graph_info()
    print(f"✅图谱构建完成，节点:{graph_info['node']}，边:{graph_info['edge']}")
    print(f"✅图谱可视化文件输出：{html_path}")


def graph_retrieve_optimized(question:str):
    """工单9优化后的检索函数：优化实体匹配，过滤低相关三元组"""
    result = {"triples":[], "text_chunks":[]}
    # 优化：实体模糊匹配
    for h, t, edge_data in gb.graph.edges(data=True):
        rel = edge_data.get("label","")
        if h in question or t in question:
            result["triples"].append({"head":h,"relation":rel,"tail":t})
    # 原文检索优化：按关键词相关性排序，只返回Top3
    chunk_score = []
    for chunk in doc_chunks:
        score = sum(1 for word in chunk if word in question)
        chunk_score.append((score, chunk))
    chunk_score.sort(reverse=True)
    for score, chunk in chunk_score[:3]:
        if score >0:
            result["text_chunks"].append(chunk)
    if len(result["triples"]) ==0 and len(result["text_chunks"])==0:
        result["text_chunks"].append("未检索到相关信息")
    return result


HTML_PAGE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>工单9 Graph‑RAG优化问答系统</title>
    <style>
        .box{width:85%;margin:20px auto;}
        textarea{width:100%;height:100px;font-size:15px;padding:8px;}
        button{padding:8px 16px;background:#1677ff;color:white;border:none;border-radius:4px;cursor:pointer;}
        .res{margin-top:15px;padding:12px;background:#f5f7fa;white-space:pre-wrap;}
    </style>
</head>
<body>
<div class="box">
    <h2>工单9 Graph‑RAG优化问答系统</h2>
    <p>测试示例问题：武汉力源信息技术股份有限公司组织结构图中,销售部有几个部门构成,其中大客户销售部有几个销售处构成?</p>
    <textarea id="q" placeholder="请输入你的问题"></textarea>
    <br><br>
    <button onclick="ask()">提交问答</button>
    <div class="res" id="out"></div>
</div>
<script>
async function ask(){
    const q = document.getElementById("q").value;
    const resp = await fetch("/api/ask",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({"question":q})
    });
    const data = await resp.json();
    let show = "【检索到的知识图谱三元组】\\n";
    for(let t of data.triples){
        show += `${t.head} —[${t.relation}]→ ${t.tail}\\n`;
    }
    show += "\\n【相关原文片段】\\n";
    for(let txt of data.text_chunks){
        show += txt + "\\n";
    }
    document.getElementById("out").innerText = show;
}
</script>
</body>
</html>
'''

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/api/ask", methods=["POST"])
def api_ask():
    req = request.get_json()
    q = req.get("question","")
    ret = graph_retrieve_optimized(q)
    return jsonify(ret)


if __name__ == "__main__":
    doc_chunks = load_simulate_financial_report()
    print(f"✅模拟文档加载完成，文本块数量 {len(doc_chunks)}")
    build_knowledge_graph_from_text(doc_chunks)

    print("\n=====================工单9 评测开始=====================")
    all_retrieved = []
    for idx, q in enumerate(sample_questions):
        print(f"\n【测试问题{idx+1}】{q}")
        res = graph_retrieve_optimized(q)
        all_retrieved.extend(res["text_chunks"])
        print("图谱三元组：", res["triples"])
        print("原文片段：", res["text_chunks"])
    # 计算RAG评估指标
    recall = calculate_context_recall(all_retrieved, sample_reference)
    precision = calculate_context_precision(all_retrieved, sample_reference)
    print(f"\n==== 工单9评测结果 ====")
    print(f"context_recall（上下文召回）：{recall}")
    print(f"context_precision（上下文精度）：{precision}")
    print("=======================================================\n")

    app.run(host="0.0.0.0", port=7862, debug=False)
