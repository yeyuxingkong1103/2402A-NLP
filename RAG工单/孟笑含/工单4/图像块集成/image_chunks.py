# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
模块：图像块提取（Qwen-VL 结果 → 知识库块）
功能：把图像解析结果加入 RAG 知识库
"""

import os
import json

# 预解析的图像内容（来自 Qwen2.5-VL-3B 解析）
IMAGE_CHUNKS = [
    {
        "chunk_id": "image_p39_org",
        "content": """[图像解析 招股说明书2.pdf 第39页 组织结构图]
根据组织结构图，武汉力源信息技术股份有限公司销售部下设4个部门：
1. 渠道销售部
2. 电话及网络销售部
3. 大客户销售部
4. 国际贸易部

大客户销售部下设6个销售处：
1. 北京销售处
2. 深圳销售处
3. 广州销售处
4. 成都销售处
5. 珠海销售处
6. 武汉销售处""",
        "page": 39,
        "doc": "招股说明书2.pdf",
        "type": "image",
        "image_type": "组织结构图",
    },
    {
        "chunk_id": "image_p72_ic",
        "content": """[图像解析 招股说明书2.pdf 第72页 2008年中国IC市场应用结构与增长图]
根据图表数据：
1) 增长率最快的行业：汽车（增长率约14.0%）
2) 负增长的行业：IC卡（增长率约-2.0%）
其他行业：工业控制（7%，增长率10.5%，位列第二）""",
        "page": 72,
        "doc": "招股说明书2.pdf",
        "type": "image",
        "image_type": "市场增长柱状图",
    },
]


def get_image_chunks():
    """返回图像块列表"""
    return IMAGE_CHUNKS


if __name__ == "__main__":
    chunks = get_image_chunks()
    print(f"✅ 共 {len(chunks)} 个图像块")
    for c in chunks:
        print(f"\n【{c['chunk_id']}】")
        print(f"  文档：{c['doc']} 第{c['page']}页")
        print(f"  类型：{c['image_type']}")
        print(f"  内容：{c['content'][:100]}...")
