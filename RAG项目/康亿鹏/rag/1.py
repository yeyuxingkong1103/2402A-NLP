# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
#
# model_path=r"D:\models\bge-reranker-large"
# model=HuggingFaceCrossEncoder(
#     model_name=model_path,
#     model_kwargs={"device":"cuda"}
# )
#
# query="什么是重排序模型"
# documents=[
#     "重排序模型广泛应用于搜索引擎和推荐系统，按相关性对候选文本进行排序",
#     "量子计算是计算科学的前沿领域",
#     "预训练语言模型的发展为重排序模型带来了新的进展"
# ]
#
# list1=[[query,i] for i in documents]
#
# score=model.score(list1)
# score=[float(i) for i in score]
# print(score)

# from pymilvus import MilvusClient
# client=MilvusClient(uri="http://localhost:19530")
# print(client.describe_collection('rag_medical'))

# from rank_bm25 import BM25Okapi
# import jieba
#
# text=['今天天气好','明天会下雨','苹果很好吃','今天星期四','明天放假',]
# token=[jieba.lcut(i) for i in text]
#
# bm25=BM25Okapi(token)
# score=bm25.get_scores(jieba.lcut('星期六放假'))
# print(score)

# import redis
#
# r=redis.Redis(
#     host='localhost',
#     port=6379,
#     db=0,
#     password=None,
#     decode_responses=True
# )
# print(r.ping())

# a=['你好','好的','我好']
# b='\n'.join(a)
# print(b)



import time
print(time.time())
print(time.perf_counter())






















