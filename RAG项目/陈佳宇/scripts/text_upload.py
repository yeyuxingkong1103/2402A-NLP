from vector_store_milvus import MilvusVectorStore

if __name__ == "__main__":
    # 实例化向量库，传入集合名
    vs = MilvusVectorStore("alice_knowledge")
    vs.create_collection()

    # 测试知识库文本，Alice角色资料
    docs = [
        "Alice今年20岁，性格温柔，喜欢星空和诗歌。",
        "Alice平时喜欢在夜晚看星星，擅长写短句小诗。",
        "Alice不喜欢吵闹的环境，偏爱安静。"
    ]
    vs.insert_texts(docs)
    print("✅ 文本入库完成！")
