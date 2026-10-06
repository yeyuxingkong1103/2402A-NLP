"""RAG 项目入口。  # 模块说明

用法:  # 命令行用法说明
    python main.py ingest                # 把 docs/ 下的文档写入 Milvus
    python main.py ingest --dir ./pdfs   # 指定目录
    python main.py ask "什么是RAG"       # 单次问答
    python main.py chat                  # 交互式问答
    python main.py search "什么是RAG"    # 查看召回+重排结果
    python main.py reset                 # 清空向量集合
    python main.py serve                 # 启动 FastAPI 服务（多用户多角色）
"""
import argparse  # 命令行参数解析库
import logging  # 运行日志
from logging.handlers import RotatingFileHandler  # 按大小滚动的文件日志

import config  # 全局配置


def cmd_ingest(args):  # 子命令 ingest：文档入库
    from loaders import SUPPORTED_EXTS, load_directory, split_documents  # 懒加载文档处理模块
    from vector_store import get_vector_store  # 懒加载向量库（避免无关命令也加载模型）

    docs = load_directory(args.dir)  # 加载指定目录下所有支持的文档
    if not docs:  # 没有加载到任何内容时
        print(f"目录 {args.dir} 下没有可加载的文档（支持: {sorted(SUPPORTED_EXTS)}）")  # 提示支持的格式
        return  # 直接结束
    chunks = split_documents(docs)  # 把文档切分成片段
    print(f"共加载 {len(docs)} 个文档，切分为 {len(chunks)} 个片段，开始写入 Milvus ...")  # 打印处理摘要

    vs = get_vector_store(args.domain)  # 获取对应领域集合的 Milvus 向量库实例（domain 必填）
    batch_size = 64  # 每批写入的片段数
    for i in range(0, len(chunks), batch_size):  # 分批写入，避免单次请求过大
        vs.add_documents(chunks[i : i + batch_size])  # 写入当前批次的片段
        print(f"  进度: {min(i + batch_size, len(chunks))}/{len(chunks)}")  # 打印写入进度
    print("写入完成。")  # 全部写入完成


def print_history(domain):  # 在终端打印指定领域的最近历史对话
    from chat_store import get_history  # 懒加载 Redis 历史读取

    history = get_history(None, domain)  # 取最近 N 轮历史（命令行模式 user_id=None）
    if not history:  # 没有历史不显示
        return
    print(f"【最近对话（领域: {domain}）】")  # 历史区标题
    for msg in history:  # 逐条打印
        print(f"{msg['role']}: {msg['content']}")  # 输出一条消息
    print("-" * 40)  # 分隔线，区分历史和本次回答


def cmd_ask(args):  # 子命令 ask：单次问答（不带问题则进入交互模式）
    from rag_chain import get_rag_chain  # 懒加载 RAG 链

    chain = get_rag_chain(args.domain, None, args.persona)  # 构建 RAG 链（领域 + 可选角色人设）
    domain = args.domain  # 领域参数（必填）
    if args.question:  # 命令行带了问题
        if domain:  # 指定了领域，先打印历史对话
            print_history(domain)  # 显示 Redis 中该领域的最近对话
        answer = ""  # 收集完整答案，用于存入历史
        for chunk in chain.stream(args.question):  # 流式生成答案
            print(chunk, end="", flush=True)  # 逐段打印、不换行、立即刷新
            answer += chunk  # 拼接答案片段
        print()  # 答案结束后补一个换行
        if domain:  # 指定了领域时把本轮问答存入 Redis 历史
            from chat_store import add_message  # 懒加载 Redis 历史存储
            add_message(None, domain, "user", args.question)  # 存用户问题（命令行 user_id=None）
            add_message(None, domain, "assistant", answer)  # 存助手回答
    else:  # 未带问题则进入交互循环
        chat_loop(chain, domain)  # 调用交互循环


def cmd_chat(args):  # 子命令 chat：交互式问答
    from rag_chain import get_rag_chain  # 懒加载 RAG 链

    domain = args.domain  # 领域参数（必填）
    chat_loop(get_rag_chain(domain, None, args.persona), domain)  # 构建链（含可选角色人设）并进入问答循环


def chat_loop(chain, domain=None):  # 交互式问答循环（ask/chat 共用）
    domain_label = domain or "默认"  # 领域显示名
    print(f"交互模式（领域: {domain_label}）：输入问题开始问答，输入 exit / quit 退出。")  # 提示使用方法
    if domain:  # 指定领域，进入时先打印历史对话
        print_history(domain)  # 显示 Redis 中该领域的最近对话
    from chat_store import add_message  # 懒加载 Redis 历史存储
    while True:  # 持续读取用户输入
        try:  # 处理 Ctrl+C / Ctrl+D 退出
            question = input("\n问题> ").strip()  # 读取问题并去首尾空白
        except (KeyboardInterrupt, EOFError):  # 用户中断
            print()  # 换行美化输出
            return  # 退出循环
        if not question:  # 空输入
            continue  # 重新提示输入
        if question.lower() in {"exit", "quit", "q"}:  # 退出命令
            return  # 结束循环
        answer = ""  # 收集完整答案，用于存入历史
        for chunk in chain.stream(question):  # 流式生成并打印答案
            print(chunk, end="", flush=True)  # 逐段输出
            answer += chunk  # 拼接答案片段
        print()  # 答案结束后换行
        if domain:  # 只有指定领域时才把对话存入 Redis 历史
            add_message(None, domain, "user", question)  # 存用户问题（命令行 user_id=None）
            add_message(None, domain, "assistant", answer)  # 存助手回答


def cmd_search(args):  # 子命令 search：只看检索结果，不生成答案
    from rag_chain import retrieve  # 懒加载检索函数

    if args.show_raw:  # 先打印两路原始召回（未融合、未重排），便于对比两路各自捞到了什么
        from hybrid_search import bm25_search  # 懒加载 BM25 召回
        from rag_chain import collection_name_for  # 懒加载领域->集合名映射
        from vector_store import get_vector_store  # 懒加载向量库（与 retrieve 复用同一缓存实例）

        vs = get_vector_store(args.domain)  # 该领域对应的 Milvus 向量库
        dense = vs.similarity_search_with_score(args.question, k=config.retrieve_k)  # 稠密召回（附带相似度）
        sparse = bm25_search(collection_name_for(args.domain), args.question, config.sparse_k)  # BM25 召回（附带 bm25_score）
        print(f"=== ① 稠密向量召回 top{len(dense)}（COSINE 相似度，越大越相关）===")  # 分节标题
        for i, (doc, score) in enumerate(dense, start=1):  # 逐条打印稠密结果
            print(f"[{i}] score={float(score):.4f} 来源={doc.metadata.get('source', '?')}")  # 相似度 + 来源
            print("    " + doc.page_content[:120].replace("\n", " "))  # 正文前 120 字（压掉换行）
        print(f"\n=== ② BM25 稀疏召回 top{len(sparse)}（bm25_score，越大越相关）===")  # 分节标题
        for i, doc in enumerate(sparse, start=1):  # 逐条打印 BM25 结果
            print(f"[{i}] score={float(doc.metadata.get('bm25_score', 0)):.4f} 来源={doc.metadata.get('source', '?')}")  # BM25 分 + 来源
            print("    " + doc.page_content[:120].replace("\n", " "))  # 正文前 120 字（压掉换行）
        print()  # 与下面的最终结果分隔

    docs = retrieve(args.question, args.domain)  # 执行"混合召回 + 重排"得到文档
    if not docs:  # 没有检索到内容
        print("没有检索到相关内容。")  # 提示无结果
        return  # 结束
    for i, doc in enumerate(docs, start=1):  # 逐条打印检索结果
        source = doc.metadata.get("source", "unknown")  # 片段来源文件路径
        score = doc.metadata.get("rerank_score")  # 重排相关性得分
        score_str = f"{float(score):.3f}" if score is not None else "-"  # 格式化得分（无则为 -）
        print(f"\n[{i}] 相关度={score_str} 来源={source}")  # 打印序号、相关度、来源
        print(doc.page_content[:500])  # 打印片段正文前 500 字


def cmd_reset(args):  # 子命令 reset：清空向量集合
    from pymilvus import MilvusClient  # Milvus 官方客户端

    domain = args.domain  # 领域参数（必填）
    collection_name = config.domain_collections[domain]  # 领域 -> 集合名
    client = MilvusClient(uri=config.milvus_uri)  # 连接 Milvus 服务
    if client.has_collection(collection_name):  # 集合存在才需要删除
        client.drop_collection(collection_name)  # 删除整个集合
        print(f"已删除集合: {collection_name}")  # 打印删除结果
    else:  # 集合不存在
        print(f"集合 {collection_name} 不存在，无需清理。")  # 提示无需操作


def cmd_serve(args):  # 子命令 serve：启动 FastAPI 服务
    import uvicorn  # 懒加载 uvicorn
    from database import init_db  # 懒加载数据库初始化

    init_db()  # 首次启动自动建表
    print(f"API 服务启动（监听 {args.host}:{args.port}）")  # 0.0.0.0 是绑定地址，不能直接在浏览器访问
    print(f"本机访问：http://localhost:{args.port}")  # 浏览器访问地址
    print(f"API 文档：http://localhost:{args.port}/docs")  # Swagger 文档地址
    uvicorn.run("api:app", host=args.host, port=args.port, log_config=None)  # log_config=None：跳过 uvicorn 自建 handler，日志统一走根 logger（含访问日志）


def main():  # 入口函数：定义并分发子命令
    if not logging.getLogger().handlers:  # 只在根 logger 未配置时配置，避免重复挂 handler、重复打开日志文件
        log_dir = config.base_dir / "logs"  # 日志目录（基于项目根目录，不受运行时 cwd 影响）
        log_dir.mkdir(parents=True, exist_ok=True)  # 目录必须先存在，否则文件 handler 构造会直接报错
        logging.basicConfig(  # 同时输出到文件和控制台
            level=logging.INFO,  # 默认 INFO：请求入口、耗时、业务失败都能看到
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # 时间 + 级别 + 模块 + 内容
            handlers=[
                RotatingFileHandler(  # 文件：写满 5MB 自动滚动，最多保留 5 个备份
                    log_dir / "rag.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8",
                ),
                logging.StreamHandler(),  # 控制台：保持终端实时可见
            ],
        )
    logging.getLogger("jieba").setLevel(logging.WARNING)  # 抑制第三方库的 DEBUG 噪音（BM25 分词会刷屏）
    parser = argparse.ArgumentParser(  # 创建参数解析器
        description="LangChain + Milvus + BGE-M3 + BGE-Reranker + DeepSeek RAG"  # 描述信息
    )
    sub = parser.add_subparsers(dest="command", required=True)  # 必须提供一个子命令

    p = sub.add_parser("ingest", help="加载文档并写入 Milvus")  # 注册 ingest 子命令
    p.add_argument("--dir", required=True, help="文档目录（必填）")  # 必填的文档目录参数
    p.add_argument("--domain", required=True, choices=list(config.domain_collections.keys()), help="选择领域（medical/education，必填）")  # 必填的领域参数
    p.set_defaults(func=cmd_ingest)  # 绑定处理函数

    p = sub.add_parser("ask", help="单次问答（不带问题则进入交互模式）")  # 注册 ask 子命令
    p.add_argument("question", nargs="?", default=None)  # 问题为可选位置参数
    p.add_argument("--domain", required=True, choices=list(config.domain_collections.keys()), help="选择领域（medical/education，必填）")  # 领域参数（必填）
    p.add_argument("--persona", default=None, help="角色人设（可选，注入 system prompt 决定回答风格）")  # 角色人设（可选）
    p.set_defaults(func=cmd_ask)  # 绑定处理函数

    p = sub.add_parser("chat", help="交互式问答")  # 注册 chat 子命令
    p.add_argument("--domain", required=True, choices=list(config.domain_collections.keys()), help="选择领域（medical/education，必填）")  # 领域参数（必填）
    p.add_argument("--persona", default=None, help="角色人设（可选，注入 system prompt 决定回答风格）")  # 角色人设（可选）
    p.set_defaults(func=cmd_chat)  # 绑定处理函数

    p = sub.add_parser("search", help="查看召回与重排结果")  # 注册 search 子命令
    p.add_argument("question")  # 必填的问题参数
    p.add_argument("--domain", required=True, choices=list(config.domain_collections.keys()), help="选择领域（medical/education，必填）")  # 领域参数（必填）
    p.add_argument("--show-raw", action="store_true", help="额外打印稠密/BM25 两路原始召回及分数（检索调参用）")  # 可选：展示原始召回
    p.set_defaults(func=cmd_search)  # 绑定处理函数

    p = sub.add_parser("reset", help="删除向量集合")  # 注册 reset 子命令
    p.add_argument("--domain", required=True, choices=list(config.domain_collections.keys()), help="选择领域（medical/education，必填）")  # 领域参数（必填）
    p.set_defaults(func=cmd_reset)  # 绑定处理函数

    p = sub.add_parser("serve", help="启动 FastAPI 服务")  # 注册 serve 子命令
    p.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")  # 主机参数
    p.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")  # 端口参数
    p.set_defaults(func=cmd_serve)  # 绑定处理函数

    args = parser.parse_args()  # 解析命令行参数
    args.func(args)  # 调用对应子命令的处理函数


if __name__ == "__main__":  # 仅直接运行本文件时执行入口
    main()  # 调用入口函数
