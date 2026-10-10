# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""16 道题的标准答案与出处标注。

答案要点（answer_key）用于计算「要点覆盖率」；
gold_pages 用于计算检索指标（命中判定：召回块页码 ∈ gold_pages），页码均为 1-based。

建库后已逐题对照语料原文复核（`_scratch/t16_verify_all.txt` 留有证据）：
凡工单初稿标注「建库后校正」的数值/项目/页码，均按原文改写并写明依据。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuestion:
    qid: int
    question: str
    question_en: str
    doc_id: str
    answer_key: list[str]
    gold_pages: list[int]
    block_type: str = "text"
    strict: bool = False
    note: str = ""


QUESTIONS: list[EvalQuestion] = [
    # ---- 武汉力源（招股说明书2）----
    EvalQuestion(
        qid=1,
        question="武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
        question_en=("How many shares are being issued by Wuhan P&S Information Technology "
                     "Co., Ltd., and what percentage of the post-issuance total share capital "
                     "does that represent?"),
        doc_id="招股说明书2",
        answer_key=["1,670万股", "占发行后总股本", "25.04%"],
        gold_pages=[2, 22, 24, 306],
        note="建库后按原文校正（工单初稿的 2,000万股/25% 与语料不符）："
             "p2/p22/p24「发行股数1,670万股，占发行后总股本的比例为25.04%」，"
             "p306 复现；p1/p3/p23 无该数据，已从 gold_pages 移除",
    ),
    EvalQuestion(
        qid=2,
        question="武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
        question_en=("Which projects does Wuhan P&S Information Technology Co., Ltd. plan to "
                     "invest the raised funds in?"),
        doc_id="招股说明书2",
        answer_key=["仓储及物流中心", "研发中心", "电子商务平台",
                    "扩充产品种类和数量", "其他与主营业务相关的营运资金"],
        gold_pages=[22, 30, 306],
        note="建库后按原文校正：p22/p306 列全部 5 个项目，p30 复述项目名；"
             "工单初稿 gold_pages [23,24,25] 无项目清单，已移除",
    ),
    EvalQuestion(
        qid=3,
        question=("与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，"
                  "持股比例和本公司关系是什么？"),
        question_en=("Who are the related parties with a controlling relationship to Wuhan P&S "
                     "Information Technology Co., Ltd., and what are their shareholding "
                     "percentages and relationship to the company?"),
        doc_id="招股说明书2",
        answer_key=["赵马克", "42.35%", "控股股东", "实际控制人"],
        gold_pages=[21, 33, 157],
        note="建库后校正：p157「赵马克 42.35% 公司控股股东」（存在控制关系的关联方表），"
             "p21/p33 载明其为控股股东及实际控制人 42.35%；"
             "要点「持股比例」改为实际数值 42.35%，工单初稿 gold_pages [36,37,38] 无此表",
    ),
    EvalQuestion(
        qid=4,
        question="与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
        question_en=("Which related-party enterprises do NOT have a controlling relationship "
                     "with Wuhan P&S Information Technology Co., Ltd.?"),
        doc_id="招股说明书2",
        answer_key=["融冰投资", "听音投资", "联众聚源", "武汉博润", "上海博润",
                    "力源贸易", "普芯达"],
        gold_pages=[157],
        note="建库后按原文校正：p157「2、不存在控制关系的关联方」共 7 家；"
             "工单初稿 gold_pages [37,38,39,40] 无此清单",
    ),
    EvalQuestion(
        qid=5,
        question=("武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，"
                  "其中大客户销售部有几个销售处构成？"),
        question_en=("In the organizational chart of Wuhan P&S Information Technology Co., Ltd., "
                     "how many departments make up the Sales Department, and how many sales "
                     "offices make up the Key Account Sales Department?"),
        doc_id="招股说明书2",
        answer_key=["渠道销售部", "电话及网络销售部", "大客户销售部", "国际贸易部",
                    "北京销售处", "深圳销售处", "广州销售处", "成都销售处",
                    "珠海销售处", "武汉销售处"],
        gold_pages=[39],
        block_type="image",
        strict=True,
        note="★图像题。组织结构图在 1-based 第 39 页（印刷页 38，idx 38）。"
             "工单P7标准答案：4个部门 + 6个销售处，10个要点必须全中。"
             "已用 data/figures/招股说明书2_p39_7fb46462e43d.png 的 VLM 结构化描述核对 10/10",
    ),
    EvalQuestion(
        qid=6,
        question=("武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与"
                  "增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？"),
        question_en=("According to the 2008 China IC market application structure and growth "
                     "chart in the prospectus of Wuhan P&S Information Technology Co., Ltd., "
                     "which industry had the fastest growth rate and which had negative growth?"),
        doc_id="招股说明书2",
        answer_key=["汽车", "14.0%", "IC卡", "-2.0%", "负增长"],
        gold_pages=[72, 310],
        block_type="image",
        note="★图像题。IC 市场图在 1-based 第 72 页（印刷页 71），第 310 页复现，"
             "两页正文均含图题「2008年中国IC市场应用结构与增长(亿元)」。"
             "已用 data/figures/招股说明书2_p72_00eb1d43b4c6.png 的 VLM 描述核对："
             "汽车14.0%增长最快；IC卡-2.0%为负增长",
    ),
    # ---- 武汉兴图新科（招股说明书1）----
    EvalQuestion(
        qid=260,
        question="报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
        question_en=("During the reporting period, what were the revenues of Wuhan Xingtu Xinke "
                     "Electronics Co., Ltd. from the military field in each period?"),
        doc_id="招股说明书1",
        answer_key=["军用领域", "6,464.51", "14,414.16", "18,780.67", "4,627.14"],
        gold_pages=[129, 130],
        note="建库后按原文校正：p129「公司来自军用领域的收入分别为6,464.51万元、"
             "14,414.16万元、18,780.67万元和4,627.14万元」（2016/2017/2018/2019H1）",
    ),
    EvalQuestion(
        qid=95,
        question="武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
        question_en=("Which technical standard did Wuhan Xingtu Xinke Electronics Co., Ltd. "
                     "participate in formulating?"),
        doc_id="招股说明书1",
        answer_key=["视频指挥系统", "技术标准", "视频技术规范"],
        gold_pages=[26, 27],
        note="实测定位：p26/p27「参与制定了全军第一个视频指挥系统技术标准"
             "（即《某视频技术规范1.0》）」",
    ),
    EvalQuestion(
        qid=33,
        question=("报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的"
                  "比重分别是多少？"),
        question_en=("During the reporting period, what proportion of main business revenue did "
                     "military-field revenue represent for Wuhan Xingtu Xinke Electronics "
                     "Co., Ltd.?"),
        doc_id="招股说明书1",
        answer_key=["82.10%", "97.31%", "94.84%", "94.34%"],
        gold_pages=[129, 130],
        note="实测定位：p129「占主营业务收入比重分别为82.10%、97.31%、94.84%和94.34%」",
    ),
    EvalQuestion(
        qid=34,
        question="根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
        question_en=("According to the prospectus of Wuhan Xingtu Xinke Electronics Co., Ltd., "
                     "which enterprises are involved in the upstream of the electronic "
                     "information industry?"),
        doc_id="招股说明书1",
        answer_key=["电子元器件制造企业", "机箱", "机柜", "金属壳体制造企业"],
        gold_pages=[152, 153],
        note="建库后按原文校正：p152「电子信息行业的上游涉及信息系统相关的电子元器件制造"
             "企业，以及机箱、机柜等金属壳体制造企业」，p153 为上下游示意图",
    ),
    EvalQuestion(
        qid=957,
        question="武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
        question_en=("In which field has Wuhan Xingtu Xinke Electronics Co., Ltd. become an "
                     "important supplier?"),
        doc_id="招股说明书1",
        answer_key=["视频指挥", "重要供应商"],
        gold_pages=[26, 27, 95],
        note="实测定位：p26「已成为国防军队视频指挥领域的重要供应商」，"
             "p27/p95「已经成为军队视频指挥领域的重要供应商」",
    ),
    EvalQuestion(
        qid=793,
        question=("根据武汉兴图新科电子股份有限公司招股意向书，"
                  "电子信息行业的下游主要包括哪些行业？"),
        question_en=("According to the prospectus of Wuhan Xingtu Xinke Electronics Co., Ltd., "
                     "what industries mainly constitute the downstream of the electronic "
                     "information industry?"),
        doc_id="招股说明书1",
        answer_key=["军队", "政府", "能源"],
        gold_pages=[152, 153],
        note="建库后按原文校正：p152「下游行业为各类终端用户……主要包括军队、政府机关、"
             "能源等行业企业」；要点「国防」非该清单原文，改为「能源」",
    ),
    EvalQuestion(
        qid=795,
        question="武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
        question_en=("Which project that Wuhan Xingtu Xinke Electronics Co., Ltd. participated "
                     "in won the First Prize of the National Science and Technology Progress Award?"),
        doc_id="招股说明书1",
        answer_key=["情报", "通信网络", "一体化工程", "C4ISR", "国家科技进步一等奖"],
        gold_pages=[27, 94, 96],
        note="实测定位：p27「某情报、指挥、控制与通信网络一体化工程（即相当于美军的C4ISR"
             "系统）荣获国家科技进步一等奖」。要点「情报指挥」改为「情报」：原文用顿号分隔"
             "（情报、指挥），照抄原文时「情报指挥」无法命中",
    ),
    EvalQuestion(
        qid=543,
        question="武汉兴图新科电子股份有限公司注册资本是多少？",
        question_en=("What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?"),
        doc_id="招股说明书1",
        answer_key=["5,520"],
        gold_pages=[22, 52],
        note="实测定位：p22「注册资本5,520.00万元」，p52「注册资本：5,520万元」。"
             "原要点 5520 与 5,520 归一化后重复、万元 无区分度，合并为单一数值要点",
    ),
    EvalQuestion(
        qid=531,
        question="武汉兴图新科电子股份有限公司法定代表人是谁？",
        question_en=("Who is the legal representative of Wuhan Xingtu Xinke Electronics Co., Ltd.?"),
        doc_id="招股说明书1",
        answer_key=["程家明"],
        gold_pages=[22, 52],
        note="实测定位：p22/p52「法定代表人：程家明」",
    ),
    EvalQuestion(
        qid=207,
        question=("武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"),
        question_en=("How much of the raised funds does Wuhan Xingtu Xinke Electronics Co., Ltd. "
                     "plan to use to supplement working capital?"),
        doc_id="招股说明书1",
        answer_key=["15,000", "补充流动资金"],
        gold_pages=[23, 30, 479],
        note="实测定位：p30/p479「补充流动资金 15,000.00 万元」，p23 募投项目表含该项目。"
             "原要点 15000 与 15,000 归一化后重复，已合并",
    ),
]

_BY_ID = {q.qid: q for q in QUESTIONS}


def get_question(qid: int) -> EvalQuestion:
    return _BY_ID[qid]


def coverage(answer: str, q: EvalQuestion) -> float:
    """答案要点覆盖率 = 命中的要点数 / 总要点数。"""
    if not answer:
        return 0.0
    a = answer.replace(" ", "").replace(",", "").replace("，", "")
    hit = 0
    for key in q.answer_key:
        k = key.replace(" ", "").replace(",", "").replace("，", "")
        if k in a:
            hit += 1
    return hit / len(q.answer_key) if q.answer_key else 0.0
