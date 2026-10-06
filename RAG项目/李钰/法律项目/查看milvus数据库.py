from pymilvus import MilvusClient

client = MilvusClient(r"C:\Users\ETERNITY\AppData\Local\Temp\milvus_legal_rag.db")

# 1. 查看所有集合
print("集合列表:", client.list_collections())

# 2. 查看数据量
stats = client.get_collection_stats("legal_cases_rag")
print("数据量:", stats)

# 3. ✅ 关键：加载集合到内存
client.load_collection("legal_cases_rag")

# 4. 再查询
results = client.query(
    collection_name="legal_cases_rag",
    filter="id > 0",
    output_fields=["text", "source"],
    limit=5
)
for i, record in enumerate(results):
    print(f"--- 第 {i+1} 条 ---")
    print(record)