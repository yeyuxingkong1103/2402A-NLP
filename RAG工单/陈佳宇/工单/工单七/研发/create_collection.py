from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
from pymilvus import utility

COLLECTION_NAME = "rag_workorder5"
DIM = 128

connections.connect("default", host="localhost", port="19530")

# 定义字段：增加主键 id！！
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), # 自增主键
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000)
]
schema = CollectionSchema(fields, description="RAG工单5集合")

# 如果集合已经存在先删除重建
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

coll = Collection(COLLECTION_NAME, schema)

# 创建索引
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
coll.create_index(field_name="vector", index_params=index_params)
coll.load()
print("✅ Milvus集合创建完成（带主键）")
