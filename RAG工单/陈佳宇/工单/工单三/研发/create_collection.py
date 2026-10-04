from pymilvus import MilvusClient, DataType

# 连接本地Milvus
client = MilvusClient(uri="http://localhost:19530")
coll_name = "rag_waterwork3"

# 如果集合存在，先删除
if client.has_collection(collection_name=coll_name):
    client.drop_collection(collection_name=coll_name)
    print(f"✅ 旧集合 {coll_name} 已删除")

# 定义schema，开启自动主键
schema = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=False,
)
# 字段
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=384)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=4096)

# 索引配置
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="IVF_FLAT",
    metric_type="L2",
    params={"nlist": 128}
)

# 创建集合
client.create_collection(
    collection_name=coll_name,
    schema=schema,
    index_params=index_params
)

# 加载集合
client.load_collection(collection_name=coll_name)
print(f"✅ 集合 {coll_name} 创建完成并加载！auto_id开启，自动生成主键id")
