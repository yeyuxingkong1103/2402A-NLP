# -*- coding: utf-8 -*-
# pdf_loader.py 工单8｜模拟金融研报加载（无外部PDF文件）
# 工单编号：人工智能NLP-RAG-基于Graph RAG实现金融问答

def load_simulate_financial_report():
    """模拟ccf_competition金融研报，直接返回文档文本，不需要pdf文件"""
    doc_text = """
武汉力源信息技术股份有限公司，是国内半导体元器件分销龙头企业。
公司总部设立于湖北省武汉市。
公司下设部门：销售部、研发部、财务部、人力资源部。

销售部包含两大业务部门：大客户销售部、中小客户销售部。
大客户销售部一共设置4个销售处，分别是：华北销售处、华东销售处、华南销售处、西南销售处。
中小客户销售部下辖：线上渠道组、线下渠道组。

研发部主要负责芯片方案设计，现有员工126人。
公司主营产品：半导体芯片、电子元器件、嵌入式模块。
主要应用行业：工业控制、汽车电子、消费电子。
2024年公司营业收入26.8亿元，同比增长12.3%。
主要竞争对手：中电港、文晔科技。

汽车电子业务是公司未来重点发展方向，计划加大研发投入，拓展国内车企客户。
公司募投项目包括：半导体测试中心建设项目、研发中心升级项目。
"""
    # 按换行切分文本块
    chunks = []
    paras = doc_text.split("\n")
    for p in paras:
        p = p.strip()
        if len(p) >15:
            chunks.append(p)
    return chunks


if __name__ == "__main__":
    chunks = load_simulate_financial_report()
    print(f"模拟文档生成完成，一共 {len(chunks)} 个文本块")
    for c in chunks:
        print("-",c)
