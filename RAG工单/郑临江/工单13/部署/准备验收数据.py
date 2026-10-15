"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。各项目独立标准答案。"""
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
original=json.loads((ROOT/'工单1/测试/gold.json').read_text(encoding='utf-8'))
for row in original:row['document']='招股说明书1.pdf'
new=[
 {'id':1,'question':'武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？','answer':'1,670万股，占发行后总股本的25.04%。','pages':[22,24],'facts':['1670','25.04%']},
 {'id':2,'question':'武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？','answer':'仓储及物流中心、研发中心、电子商务平台、扩充产品种类和数量、其他与主营业务相关的营运资金。','pages':[22,30],'facts':['仓储及物流中心','研发中心','电子商务平台','扩充产品种类和数量','其他与主营业务相关的营运资金']},
 {'id':3,'question':'与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？','answer':'赵马克，持股比例42.35%，为公司控股股东。','pages':[157],'facts':['赵马克','42.35%','控股股东']},
 {'id':4,'question':'与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？','answer':'融冰投资、武汉博润、上海博润、听音投资、联众聚源（持股5%以上股东）；力源贸易（同一实际控制人控制的企业）；普芯达（实际控制人近亲属控制的公司）。','pages':[157],'facts':['融冰投资','武汉博润','上海博润','听音投资','联众聚源','力源贸易','普芯达']},
 {'id':5,'question':'武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？','answer':'销售部下设4个部门：电话及网络销售部、渠道销售部、大客户销售部、国际贸易部。大客户销售部有6个销售处：珠海、深圳、北京、武汉、广州、成都销售处。','pages':[39],'facts':['4','6','电话及网络销售部','渠道销售部','大客户销售部','国际贸易部']},
 {'id':6,'question':'武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？','answer':'汽车电子增长最快，为14.0%；IC卡为负增长，为-2.0%。','pages':[72],'facts':['汽车','14.0%','IC卡','-2.0%']},
 {'id':15,'question':'武汉力源信息技术股份有限公司法定代表人是谁？','answer':'赵马克。','pages':[15,23],'facts':['赵马克']},
 {'id':16,'question':'武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？','answer':'大客户销售部，下设6个销售处：珠海、深圳、北京、武汉、广州、成都销售处。','pages':[39],'facts':['大客户销售部','珠海','深圳','北京','武汉','广州','成都']},
]
for row in new:row['document']='招股说明书2.pdf'
for num in [2,3,4,5,6,11,12,13]:
    folder=ROOT/f'工单{num}'
    gold=original+(new[:4] if num==3 else new if num>=4 else [])
    (folder/'测试/gold.json').write_text(json.dumps(gold,ensure_ascii=False,indent=2),encoding='utf-8')
    if num in {5,6,11,12,13}:
        shutil.copytree(ROOT/'工单4/研发/data/images',folder/'研发/data/images',dirs_exist_ok=True)
        shutil.copy2(ROOT/'工单4/研发/data/image_annotations.json',folder/'研发/data/image_annotations.json')
    multi=[original[0]['question'],'他参与的哪个工程荣获了国家科技进步一等奖？','这个公司的法定代表人是谁？','那武汉力源信息技术股份有限公司呢？',new[-1]['question']]
    (folder/'测试/多轮对话用例.json').write_text(json.dumps(multi,ensure_ascii=False,indent=2),encoding='utf-8')
print('招股问题及独立图像附件已准备')
