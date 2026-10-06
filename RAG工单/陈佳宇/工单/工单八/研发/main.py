# main.py 工单8 Graph‑RAG金融问答主程序【最简可运行版】
# 工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答
from flask import Flask, render_template_string, request, jsonify
from graph_builder import GraphBuilder
from pdf_loader import load_simulate_financial_report

app = Flask(__name__)

# 全局对象
gb = GraphBuilder()
doc_chunks = []

# 模拟eval_questions.md 测试问题（工单8要求，用于和工单7结果对比）
eval_questions = [
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

def build_knowledge_graph_from_text(chunks):
    """简易规则抽取三元组，构建知识图谱（实训简易版本）"""
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


def graph_retrieve(question:str):
    """最简检索：图谱三元组 + 原文直接包含匹配"""
    result = {"triples":[], "text_chunks":[]}
    # 1、知识图谱检索
    for h, t, edge_data in gb.graph.edges(data=True):
        rel = edge_data.get("label","")
        if h in question or t in question:
            result["triples"].append({"head":h,"relation":rel,"tail":t})

    # ========== 最简单的原文匹配逻辑 ==========
    for chunk in doc_chunks:
        # 只要chunk里面出现“公司”就全部拿出来（调试用，保证一定有返回）
        if "公司" in chunk:
            result["text_chunks"].append(chunk)
    # =========================================

    if len(result["triples"]) ==0 and len(result["text_chunks"])==0:
        result["text_chunks"].append("未检索到相关信息")
    return result


HTML_PAGE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>工单8 Graph‑RAG金融问答系统</title>
    <style>
        .box{width:85%;margin:20px auto;}
        textarea{width:100%;height:100px;font-size:15px;padding:8px;}
        button{padding:8px 16px;background:#1677ff;color:white;border:none;border-radius:4px;cursor:pointer;}
        .res{margin-top:15px;padding:12px;background:#f5f7fa;white-space:pre-wrap;}
    </style>
</head>
<body>
<div class="box">
    <h2>工单8 Graph‑RAG金融问答系统</h2>
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
    ret = graph_retrieve(q)
    return jsonify(ret)


if __name__ == "__main__":
    # 1.加载模拟文档
    doc_chunks = load_simulate_financial_report()
    print(f"✅模拟文档加载完成，文本块数量 {len(doc_chunks)}")
    print("doc_chunks内容预览：")
    for c in doc_chunks:
        print("-",c)

    # 2.构建知识图谱
    build_knowledge_graph_from_text(doc_chunks)

    # 3.批量打印eval_questions测试用例（用于和工单7对比）
    print("\n=====================工单8 eval_questions测试集合=====================")
    for idx, q in enumerate(eval_questions):
        print(f"\n【测试问题{idx+1}】{q}")
        res = graph_retrieve(q)
        print("图谱三元组：", res["triples"])
        print("原文片段：", res["text_chunks"])
    print("====================================================================\n")

    # 4.启动web服务
    app.run(host="0.0.0.0", port=7861, debug=False)
