"""
Milvus 连通性与数据自检脚本

用途：不启动 FastAPI 服务，直接连上 Milvus 看看「有哪些集合、每个集合多少条数据、
      前几条数据长什么样」，用来判断「检索没结果」到底是库里真没数据，还是索引/检索链路的问题。

本模块在系统中的位置：
    这是一个独立的运维小工具，没有别的模块调用它，它也不依赖项目内的其他模块（只用 pymilvus）。
    排查顺序建议：先跑本脚本确认 Milvus 里有数据，再去查 retriever 里的检索与重排逻辑。

关键取舍：
    - 全部是只读 query，不写也不删，任何时候跑都安全。
    - 地址写死在 localhost:19530（Milvus 默认端口），换机器时改这一行即可，需与 config.MILVUS_URI 保持同源。
"""
from pymilvus import MilvusClient  # 原生客户端：本脚本不需要向量模型，直接用 pymilvus 最省事

client = MilvusClient(uri='http://localhost:19530')  # 建立到 Milvus 的连接
cols = client.list_collections()  # 列出实例上所有集合名（rag_legal / rag_psychology / rag_companion 等）
print('Collections:', cols)
for c in cols:  # 逐个集合看数据量，一眼能看出哪个角色库还是空的
    count = client.get_collection_stats(c).get("row_count", 0)  # row_count 是统计值；Milvus 逻辑删除的数据可能仍被计入
    print(f'\n  {c}: {count} entities')
    if count > 0:  # 空集合不查，避免无谓报错
        try:
            rows = client.query(  # 每个集合抽样 3 条，核对主键和正文是否正常
                collection_name=c,
                filter='',  # 空表达式表示不过滤，随便取几条
                output_fields=['pk','text','source'] if c == 'persona_rag' else ['pk','text'],  # 只有老的 persona_rag 集合带 source 字段，别的集合查它会报错，所以按集合名区分字段
                limit=3  # 只抽 3 条，控制打印量
            )
            for r in rows:
                pk = str(r.get('pk','?'))[:16]  # 主键是 uuid 字符串，截前 16 位展示就够辨认
                source = r.get('source','N/A')  # 该集合没有 source 字段时显示 N/A
                text = str(r.get('text',''))[:80]  # 正文只截 80 字预览
                print(f'    pk={pk}... source={source} text={text}...')
        except Exception as e:
            print(f'    Query error: {e}')  # 单个集合查询失败不中断整体自检：打印原因后继续看下一个集合
