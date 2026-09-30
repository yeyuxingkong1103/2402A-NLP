# -*- coding: utf-8 -*-
"""单元测试模块：第 1 步覆盖 MySQL / Redis / Milvus 基础连接。"""

import os                          # 导入 os 模块，用于拼接路径
import sys                         # 导入 sys 模块，用于调整模块搜索路径
import pytest                      # 导入 pytest，用于 skip 与断言

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 把项目根目录加入搜索路径

MILVUS_REQUIRED_FIELDS = {         # Milvus Collection 必须包含的字段集合
    "id", "vector", "sparse_vector", "text", "summary", "source", "created_at", "updated_at"  # 8 个字段
}                                  # 集合定义结束

CHUNK_MIN_LEN = 80                 # 语义块最小长度，与 ingest.py 保持一致
CHUNK_MAX_LEN = 500                # 语义块最大长度，与 ingest.py 保持一致

# 测试用长文本：两段内容，合计长度足够切出多个块
SAMPLE_TEXT = (                    # 测试文本开始
    "变压器漏油是常见的设备缺陷。发现漏油后应先断电并挂设接地线，确认无残余电荷后再进行处理。"   # 第一句
    "处理步骤包括：检查油箱焊缝、更换老化的密封胶垫、按力矩要求复紧法兰螺栓。"                  # 第二句
    "若漏油点位于套管根部，应申请停电检修，禁止带电紧固。\n\n"                                 # 第三句并分段
    "绝缘油中溶解气体分析是判断变压器内部故障的重要手段。当乙炔含量超过注意值时，"              # 第四句
    "应结合三比值法判断故障类型，并及时安排停电检查。运行中应定期取样，记录各组分含量变化趋势。"  # 第五句
)                                  # 测试文本结束

# 测试用带页眉页脚与水印的文本
DIRTY_TEXT = (                     # 脏文本开始
    "GB/T27743—2025\n"             # 页眉：标准号，属于跨页重复行
    "5 通用检测方法\n"              # 正文标题
    "空载运转性能试验前,应进行设备工作环境、电压波动情况检测。\n"   # 正文内容
    "受控\n"                       # 水印短行
    "12\n"                         # 页码：纯数字
    "GB/T27743—2025"               # 页脚：标准号，与页眉重复
)                                  # 脏文本结束


def test_placeholder() -> None:
    """占位用例：确认测试框架可以正常跑通。"""
    assert True                    # 断言恒为真，仅用于验证 pytest 可用


def test_config_loads() -> None:
    """占位用例：确认配置模块可以被导入（后续步骤再补充真实断言）。"""
    pytest.importorskip("dotenv")  # 缺少 dotenv 依赖时跳过本用例
    import config                  # 导入配置模块，验证没有语法或导入错误
    assert config is not None      # 断言模块对象存在


def test_redis_demo() -> None:
    """测试 Redis 5 种数据类型演示：Redis 连不上时跳过，不判定为失败。"""
    pytest.importorskip("dotenv")  # 缺少 dotenv 依赖时跳过本用例
    pytest.importorskip("redis")   # 缺少 redis 依赖时跳过本用例
    import db                      # 导入 db 模块（内部会读取 config 配置）
    try:                           # 先探测 Redis 是否可以连通
        db.get_redis_client().ping()   # ping 成功说明服务可用
    except Exception as exc:       # 连不上（服务没启动或配置为空）
        pytest.skip(f"Redis 不可用，跳过：{exc}")   # 跳过本用例
    result = db.redis_demo()       # 执行 5 种数据类型演示
    assert result                  # 断言演示结果非空
    assert set(result.keys()) == {"string", "list", "hash", "set", "zset"}  # 断言 5 种类型齐全
    assert result["string"] == "变压器漏油处理"   # 断言 string 类型读写正确
    assert len(result["list"]) == 3               # 断言 list 类型写入了 3 条
    assert result["set"] == ["绝缘老化", "过热"] or len(result["set"]) == 2  # 断言 set 已去重
    assert "变压器" in result["hash"].values()    # 断言 hash 字段写入了设备名


def test_milvus_collection_fields() -> None:
    """测试 Milvus Collection 字段齐全：Milvus 连不上时跳过，字段校验用本地 schema。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    pytest.importorskip("pymilvus")   # 缺少 pymilvus 依赖时跳过本用例
    import vector_store               # 导入向量库模块
    field_names = set(vector_store.get_field_names())   # 取本地 schema 的字段名（无需连服务）
    missing = MILVUS_REQUIRED_FIELDS - field_names      # 计算缺少的字段
    assert not missing, f"Collection 缺少字段：{missing}"  # 断言没有任何字段缺失
    try:                              # 再尝试真实连接 Milvus
        client = vector_store.get_milvus_client()   # 获取客户端
        client.list_collections()     # 列集合，验证服务真的可用
    except Exception as exc:          # 连不上（服务没启动或配置为空）
        pytest.skip(f"Milvus 不可用，字段校验已通过，跳过连接检查：{exc}")   # 跳过连接部分


def test_semantic_chunk_basic() -> None:
    """测试语义切分：块数至少为 1，且每块长度在 min_len 与 max_len 之间。"""
    import ingest_chunk               # 导入文本处理模块
    chunks = ingest_chunk.semantic_chunk(SAMPLE_TEXT, CHUNK_MAX_LEN, CHUNK_MIN_LEN)   # 执行切分
    assert len(chunks) >= 1, "切分结果至少应有 1 块"                   # 断言块数下限
    for chunk in chunks:              # 逐块检查长度
        assert len(chunk) >= CHUNK_MIN_LEN, f"块过短：{len(chunk)}"    # 断言不小于最小长度
        assert len(chunk) <= CHUNK_MAX_LEN, f"块过长：{len(chunk)}"    # 断言不超过最大长度
    joined = "".join(chunks)          # 把块拼回去
    for key in ("变压器漏油", "三比值法"):        # 检查关键内容没有丢失
        assert key in joined, f"切分后丢失内容：{key}"                # 断言关键内容仍在


def test_remove_watermark() -> None:
    """测试去水印与页眉页脚：清洗后不含标准号页眉、页码、水印行，正文保留。"""
    import ingest_chunk               # 导入文本处理模块
    repeated = {"GB/T27743—2025"}     # 模拟跨页重复行识别结果
    cleaned = ingest_chunk.remove_watermark_and_header_footer(DIRTY_TEXT, 5, 15, repeated)   # 执行清洗
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]   # 取清洗后的非空行
    assert "GB/T27743—2025" not in lines, "页眉页脚未被去除"            # 断言标准号页眉被去掉
    assert "12" not in lines, "页码行未被去除"                          # 断言纯数字页码被去掉
    assert "受控" not in lines, "水印行未被去除"                        # 断言水印短行被去掉
    assert any("5 通用检测方法" in ln for ln in lines), "正文标题被误删"  # 断言正文标题保留
    assert any("空载运转性能试验" in ln for ln in lines), "正文内容被误删"  # 断言正文内容保留


def test_remove_watermark_keeps_standard_word() -> None:
    """补充用例：含"标准"两字但属于正文的行不能被误删。"""
    import ingest_chunk               # 导入文本处理模块
    text = "中华人民共和国国家标准\n标准值应不超过额定电压的百分之十。"   # 含关键词的正文
    cleaned = ingest_chunk.remove_watermark_and_header_footer(text, 1, 10, set())   # 执行清洗
    assert "中华人民共和国国家标准" in cleaned, "11 字的正文标题被误删"   # 断言长标题保留
    assert "标准值应不超过" in cleaned, "含关键词的正文被误删"           # 断言正文保留


def test_load_chunks_from_json() -> None:
    """测试读取解析结果：块数大于 0，且每条都带 text / source / summary 三个字段。"""
    import vector_store               # 导入向量库模块
    chunks = vector_store.load_chunks_from_json()   # 读取并去重
    assert len(chunks) > 0, "解析结果里没有读到任何块"          # 断言块数大于 0
    for item in chunks:               # 逐条检查字段
        for key in ("text", "source", "summary"):              # 三个必需字段
            assert key in item, f"块缺少字段：{key}"            # 断言字段齐全
        assert item["text"].strip(), "存在空的块文本"           # 断言文本非空
    assert len({c["text"] for c in chunks}) == len(chunks), "去重不彻底"   # 断言无重复文本


def test_embed_texts_shape() -> None:
    """测试向量化：返回 1024 维向量；模型未装或路径无效时跳过。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    pytest.importorskip("FlagEmbedding")   # 缺少 FlagEmbedding 时跳过本用例
    import vector_store               # 导入向量库模块
    import config                     # 导入配置模块
    from pathlib import Path          # 导入 Path
    if not config.BGE_M3_PATH or not Path(config.BGE_M3_PATH).exists():   # 模型路径无效
        pytest.skip("BGE-M3 模型路径不存在，跳过")             # 跳过本用例
    dense, sparse = vector_store.embed_texts(["测试"])          # 向量化一条文本
    assert len(dense) == 1, "稠密向量条数与输入不一致"           # 断言条数一致
    assert len(dense[0]) == 1024, f"向量维度不是 1024：{len(dense[0])}"   # 断言维度
    assert len(sparse) == 1, "稀疏向量条数与输入不一致"          # 断言稀疏条数一致
    assert sparse[0], "稀疏向量不应为空"                        # 断言稀疏向量非空
    for key, value in sparse[0].items():      # 逐项检查稀疏向量格式
        assert isinstance(key, int), f"稀疏向量键必须是整数：{key}"       # Milvus 要求整数词项
        assert isinstance(value, float), f"稀疏向量值必须是浮点：{value}"   # Milvus 要求浮点权重
    assert vector_store.embed_texts([]) == ([], []), "空输入应返回两个空列表"   # 断言空输入处理


def test_milvus_row_count() -> None:
    """测试入库结果：Milvus 中行数大于 0；Milvus 连不上时跳过。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    pytest.importorskip("pymilvus")   # 缺少 pymilvus 依赖时跳过本用例
    import vector_store               # 导入向量库模块
    try:                              # 先探测 Milvus 是否可连
        vector_store.get_milvus_client().list_collections()   # 列集合验证连通性
    except Exception as exc:          # 连不上
        pytest.skip(f"Milvus 不可用，跳过：{exc}")             # 跳过本用例
    result = vector_store.verify_insert()                     # 校验入库结果
    assert result["row_count"] > 0, f"集合 {result['collection']} 行数为 0"   # 断言行数大于 0
    assert result["sample"], "取不到样例数据"                 # 断言有样例返回


def test_hybrid_search_milvus() -> None:
    """测试混合检索：retrieve("绝缘油") 至少返回 1 条；Milvus 连不上则跳过。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    pytest.importorskip("pymilvus")   # 缺少 pymilvus 依赖时跳过本用例
    import retrieval                  # 导入检索模块
    import vector_store               # 导入向量库模块
    try:                              # 先探测 Milvus 是否可连
        vector_store.get_milvus_client().list_collections()   # 列集合验证连通性
    except Exception as exc:          # 连不上
        pytest.skip(f"Milvus 不可用，跳过：{exc}")             # 跳过本用例
    results = retrieval.retrieve("绝缘油")                    # 走完整检索链路
    assert len(results) >= 1, "混合检索没有返回任何结果"       # 断言至少一条
    for item in results:              # 逐条检查结构
        for key in ("text", "source", "score"):               # 三个必备字段
            assert key in item, f"检索结果缺少字段：{key}"      # 断言字段齐全


def test_rrf_fusion() -> None:
    """测试 RRF 融合：两路都命中的文档应排在只被单路命中的文档前面。"""
    import retrieval                  # 导入检索模块
    dense_hits = [                    # 构造稠密路结果：A 第 1 名，B 第 2 名
        {"id": 1, "text": "A", "rank": 0},
        {"id": 2, "text": "B", "rank": 1},
    ]
    sparse_hits = [                   # 构造稀疏路结果：B 第 1 名，C 第 2 名
        {"id": 2, "text": "B", "rank": 0},
        {"id": 3, "text": "C", "rank": 1},
    ]
    fused = retrieval.rrf_fuse(dense_hits, sparse_hits, k=60)   # 执行融合
    assert len(fused) == 3, f"融合后应去重为 3 条，实际 {len(fused)}"   # 断言去重条数
    assert fused[0]["id"] == 2, "两路都命中的文档应排第一"      # 断言 B 排第一
    assert fused[0]["from"] == "dense+sparse", "命中路标记不正确"       # 断言命中路标记
    scores = [item["rrf_score"] for item in fused]              # 取出分数序列
    assert scores == sorted(scores, reverse=True), "融合结果未按分数降序"   # 断言降序
    expected = 1 / 61 + 1 / 62        # B 在两路分别是第 2 名和第 1 名（rank 从 0 起）
    assert abs(fused[0]["rrf_score"] - expected) < 1e-9, "RRF 公式计算不正确"   # 断言公式


def test_rerank_score_filter(monkeypatch) -> None:
    """测试精排过滤：sigmoid 分数低于阈值的候选应被丢弃。"""
    import retrieval                  # 导入检索模块
    import config                     # 导入配置模块

    class FakeReranker:               # 伪造的重排模型，用于确定性地验证过滤逻辑
        def compute_score(self, pairs, normalize=True):   # 按文档顺序返回固定分数
            table = {                 # 文档文本 → 分数映射
                "相关文档甲": 0.9,      # 高分，应保留
                "无关文档乙": 0.1,      # 低分，应被过滤
                "边缘文档丙": 0.5,      # 中间分，应保留
            }                         # 映射结束
            return [table.get(pair[1], 0.0) for pair in pairs]   # 逐条返回分数

    monkeypatch.setattr(retrieval, "load_reranker", lambda: FakeReranker())   # 替换为假模型
    docs = [                          # 构造 3 条假文档
        {"id": 1, "text": "相关文档甲"},   # 高分文档
        {"id": 2, "text": "无关文档乙"},   # 低分文档
        {"id": 3, "text": "边缘文档丙"},   # 中间分文档
    ]
    kept = retrieval.rerank("任意问题", docs, top_k=5)          # 执行精排
    texts = [item["text"] for item in kept]                     # 取出保留的文本
    assert "无关文档乙" not in texts, "低于阈值的候选未被过滤"   # 断言低分被丢弃
    assert "相关文档甲" in texts, "高分候选被误删"               # 断言高分保留
    assert "边缘文档丙" in texts, "高于阈值的候选被误删"         # 断言中间分保留
    assert len(kept) == 2, f"应保留 2 条，实际 {len(kept)}"      # 断言行数
    assert kept[0]["text"] == "相关文档甲", "未按分数降序排列"    # 断言排序
    assert config.RERANK_SCORE_THRESHOLD == 0.3, "过滤阈值应为 0.3"   # 断言阈值配置


def test_rerank_real_model() -> None:
    """集成用例：用真实 reranker 打分，断言保留项分数都不低于阈值；加载失败则跳过。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    pytest.importorskip("FlagEmbedding")   # 缺少 FlagEmbedding 时跳过本用例
    import retrieval                  # 导入检索模块
    import config                     # 导入配置模块
    from pathlib import Path          # 导入 Path
    if not config.RERANKER_PATH or not Path(config.RERANKER_PATH).exists():   # 模型路径无效
        pytest.skip("Reranker 模型路径不存在，跳过")            # 跳过本用例
    try:                              # 尝试真实加载与打分
        docs = [                      # 一条相关、一条无关
            {"id": 1, "text": "绝缘油中溶解气体的气相色谱分析方法与试验步骤"},
            {"id": 2, "text": "今天天气很好，适合出门散步和购物"},
        ]
        kept = retrieval.rerank("绝缘油气相色谱分析步骤", docs, top_k=5)   # 精排
    except Exception as exc:          # 模型加载或推理失败
        pytest.skip(f"Reranker 不可用，跳过：{exc}")            # 跳过本用例
    for item in kept:                 # 逐条检查
        assert item["rerank_score"] >= config.RERANK_SCORE_THRESHOLD, "保留了低于阈值的项"   # 断言阈值
    assert kept, "精排后不应为空"       # 断言至少保留一条


def test_multi_recall_dedup(monkeypatch) -> None:
    """测试多路召回去重：两路出现相同 text 时只保留一条。"""
    import retrieval                  # 导入检索模块
    milvus_fake = [                   # 伪造 Milvus 路结果
        {"id": 1, "text": "变压器漏油处理", "source": "milvus", "score": 0.9},
        {"id": 2, "text": "绝缘油色谱分析", "source": "milvus", "score": 0.8},
    ]
    mysql_fake = [                    # 伪造 MySQL 路结果，第一条与 Milvus 路重复
        {"id": 9, "text": "绝缘油色谱分析", "source": "mysql", "score": 0.5},
        {"id": 8, "text": "历史提问内容", "source": "mysql", "score": 0.4},
    ]
    monkeypatch.setattr(retrieval, "hybrid_search_milvus", lambda q, k=None: milvus_fake)   # 替换 Milvus 路
    monkeypatch.setattr(retrieval, "search_mysql", lambda q, k=None: mysql_fake)            # 替换 MySQL 路
    merged = retrieval.multi_recall("任意问题")                 # 执行多路召回
    texts = [item["text"] for item in merged]                   # 取出文本列表
    assert len(texts) == 3, f"去重后应为 3 条，实际 {len(texts)}"    # 断言去重条数
    assert len(set(texts)) == len(texts), "存在重复文本，去重未生效"   # 断言无重复
    assert "变压器漏油处理" in texts and "历史提问内容" in texts       # 断言两路内容都在


def test_build_messages_structure() -> None:
    """测试提示词组装：首条是 system、末条是 user，且 user 里含检索上下文与问题。"""
    import rag                        # 导入问答模块
    contexts = [                      # 构造两条检索结果
        {"source": "GBT 17623-2026", "text": "条款内容甲", "score": 0.9},
        {"source": "GBT 27743-2025", "text": "条款内容乙", "score": 0.8},
    ]
    messages = rag.build_messages("绝缘油怎么测", contexts)   # 无历史时组装
    assert len(messages) == 2, f"无历史时应为 2 条消息，实际 {len(messages)}"   # 断言条数
    assert messages[0]["role"] == "system", "首条不是 system"     # 断言首条角色
    assert messages[-1]["role"] == "user", "末条不是 user"        # 断言末条角色
    assert "电力维修专家" in messages[0]["content"], "system 不是专家角色"   # 断言角色设定
    user_text = messages[-1]["content"]                           # 取出用户消息
    assert "问题：绝缘油怎么测" in user_text, "user 消息缺少问题"   # 断言含问题
    assert "[1] 来源：GBT 17623-2026" in user_text, "user 消息缺少带编号的上下文"   # 断言含上下文
    assert "[2] 来源：GBT 27743-2025" in user_text, "第二条上下文缺失"   # 断言第二条也在
    history = [                       # 构造历史对话
        {"role": "user", "content": "上一轮问题"},
        {"role": "assistant", "content": "上一轮回答"},
    ]
    with_history = rag.build_messages("追问", contexts, history)   # 带历史组装
    roles = [m["role"] for m in with_history]                     # 取出角色序列
    assert roles == ["system", "user", "assistant", "user"], f"历史插入位置不对：{roles}"   # 断言顺序
    empty = rag.build_messages("无关问题", [])                    # 空上下文组装
    assert "没有找到任何相关条款" in empty[-1]["content"], "空上下文未给出提示"   # 断言空上下文提示


def test_postprocess_clean_blank_lines() -> None:
    """测试后处理：连续 3 个以上空行合并为 2 个，且不能把换行全吃掉。"""
    import re                         # 导入正则模块
    import rag                        # 导入问答模块
    dirty = "第一段。\n\n\n\n第二段。\n\n\n\n\n第三段。"   # 含多个连续空行
    cleaned = rag.postprocess(dirty)  # 执行后处理
    runs = re.findall(r"\n+", cleaned)                    # 找出所有连续换行
    assert runs, "段落分隔被全部删除，换行丢失"            # 断言换行还在
    assert max(len(run) for run in runs) <= 2, f"仍存在超过 2 个的连续空行：{runs}"   # 断言空行数
    for word in ("第一段", "第二段", "第三段"):            # 逐段检查内容
        assert word in cleaned, f"后处理丢内容：{word}"    # 断言内容未丢


def test_postprocess_truncate() -> None:
    """测试后处理截断：超长文本被截到 8000 字以内并带截断提示。"""
    import rag                        # 导入问答模块
    cleaned = rag.postprocess("变压器" * 3000)            # 6000 字，不触发截断
    assert len(cleaned) <= rag.MAX_ANSWER_LEN, "未超长却被截断"   # 断言不误截
    long_text = rag.postprocess("变压器" * 5000)          # 10000 字，触发截断
    assert len(long_text) <= rag.MAX_ANSWER_LEN, f"截断后仍超长：{len(long_text)}"   # 断言长度上限
    assert long_text.endswith(rag.TRUNCATE_SUFFIX), "截断后缺少提示语"   # 断言带提示
    check = rag.validate_answer(long_text)                # 截断结果应能通过校验
    assert check["ok"], f"截断结果未通过校验：{check['reason']}"   # 断言校验通过


def test_validate_answer_safety() -> None:
    """测试答案校验：含违规表述时判定不通过。"""
    import rag                        # 导入问答模块
    result = rag.validate_answer("按本方法操作绝对安全，不会出问题。")   # 含违规词
    assert not result["ok"], "含违规词却判定通过"          # 断言不通过
    assert "绝对安全" in result["reason"], f"未指出违规词：{result['reason']}"   # 断言原因含违规词
    result2 = rag.validate_answer("按此操作保证不出问题。")   # 另一个违规词
    assert not result2["ok"], "含违规词却判定通过"          # 断言不通过
    empty = rag.validate_answer("")                        # 空答案
    assert not empty["ok"], "空答案却判定通过"              # 断言不通过
    normal = rag.validate_answer("变压器应停电检修，按 GB/T 27743—2025 第 5 章执行。")   # 正常答案
    assert normal["ok"], f"正常答案被误判：{normal['reason']}"   # 断言通过


def test_validate_answer_no_evidence() -> None:
    """测试答案校验：含无依据话术时应标记 is_no_evidence=True。"""
    import rag                        # 导入问答模块
    result = rag.validate_answer("标准文档中没有找到相关条款。")   # 含无依据话术
    assert result["is_no_evidence"], "未标记无依据"        # 断言标记为真
    assert result["ok"], "无依据话术本身不应判为不通过"      # 断言仍可通过校验
    normal = rag.validate_answer("按 GB/T 27743—2025 第 5 章执行。")   # 正常答案
    assert not normal["is_no_evidence"], "正常答案被误标为无依据"   # 断言标记为假
    mixed = rag.validate_answer("标准文档中没有找到相关条款，且本操作绝对安全。")   # 无依据 + 违规词
    assert mixed["is_no_evidence"], "无依据标记丢失"        # 断言仍标记无依据
    assert not mixed["ok"], "含违规词却判定通过"            # 断言违规词优先判不通过


def test_call_deepseek_stream_mock(monkeypatch) -> None:
    """测试流式调用：mock 掉 openai 客户端，断言正文被逐段 yield 且以结束标记收尾。"""
    import rag                        # 导入问答模块

    class FakeDelta:                  # 伪造的增量对象
        def __init__(self, content):  # 构造方法
            self.content = content    # 保存正文内容

    class FakeChoice:                 # 伪造的选项对象
        def __init__(self, content):  # 构造方法
            self.delta = FakeDelta(content)   # 挂上增量

    class FakeUsage:                  # 伪造的用量对象
        prompt_tokens = 11            # 输入 token 数
        completion_tokens = 22        # 输出 token 数
        total_tokens = 33             # 合计 token 数

    class FakeChunk:                  # 伪造的流式分片
        def __init__(self, content=None, usage=None):   # 构造方法
            self.choices = [FakeChoice(content)] if content is not None else []   # 空增量时无选项
            self.usage = usage        # 用量，仅最后一片有

    class FakeCompletions:            # 伪造的补全接口
        def create(self, **kwargs):   # 接收任意参数
            assert kwargs.get("stream") is True, "未开启流式"   # 断言调用方开了流式
            return iter([             # 返回分片序列
                FakeChunk("变压"),     # 第一段
                FakeChunk(""),         # 空增量，应被跳过
                FakeChunk("器漏油"),   # 第二段
                FakeChunk(None, FakeUsage()),   # 只带用量的收尾片
            ])                        # 分片序列结束

    class FakeClient:                 # 伪造的客户端
        def __init__(self):           # 构造方法
            self.chat = type("C", (), {"completions": FakeCompletions()})()   # 挂上补全接口

    monkeypatch.setattr(rag, "_client", lambda: FakeClient())   # 替换真实客户端
    usage = {}                        # 用于接收用量
    pieces = list(rag.call_deepseek_stream([{"role": "user", "content": "x"}], usage))   # 收集所有产出
    assert pieces[-1] == rag.END_MARKER, f"未以结束标记收尾：{pieces[-1]}"   # 断言结束标记
    assert pieces[:-1] == ["变压", "器漏油"], f"正文分片不正确：{pieces[:-1]}"   # 断言空增量被跳过
    assert usage.get("total_tokens") == 33, f"用量未回传：{usage}"   # 断言用量回传

    def boom(**kwargs):               # 伪造一个必然抛异常的接口
        raise RuntimeError("模拟网络中断")   # 抛出异常

    monkeypatch.setattr(FakeCompletions, "create", boom)   # 替换成异常版本
    broken = list(rag.call_deepseek_stream([{"role": "user", "content": "x"}]))   # 再收集一次
    assert broken[0] == rag.INTERRUPT_TEXT, f"异常时未输出中断提示：{broken}"   # 断言中断提示
    assert broken[-1] == rag.END_MARKER, "异常时未以结束标记收尾"   # 断言仍给结束标记


def _mysql_ready() -> bool:           # 辅助函数：探测 MySQL 是否可用
    """探测 MySQL 是否可连，不可用时返回 False。"""
    try:                              # 尝试查询角色表
        import db_user                # 导入用户数据模块
        db_user.list_roles()          # 探测连通性
        return True                   # 可用
    except Exception:                 # 连不上
        return False                  # 不可用


def test_hash_password() -> None:
    """测试密码哈希：同一密码结果一致，不同密码结果不同，且不落明文。"""
    import db_user                    # 导入用户数据模块
    import config                     # 导入配置模块
    first = db_user.hash_password("pwd123456")            # 第一次计算
    second = db_user.hash_password("pwd123456")           # 第二次计算
    other = db_user.hash_password("pwd123457")            # 换一个密码
    assert first == second, "同一密码两次哈希结果不一致"   # 断言确定性
    assert first != other, "不同密码哈希结果相同"          # 断言区分性
    assert len(first) == 64, f"SHA256 十六进制应为 64 位：{len(first)}"   # 断言长度
    assert "pwd123456" not in first, "哈希结果里出现明文密码"   # 断言不含明文
    assert db_user.verify_password("pwd123456", first), "正确密码校验失败"   # 断言校验通过
    assert not db_user.verify_password("wrong", first), "错误密码校验通过"   # 断言校验失败
    assert config.PASSWORD_SALT, "盐值未配置"              # 断言盐值已配置


def _redis_ready() -> bool:           # 辅助函数：探测 Redis 是否可用
    """探测 Redis 是否可连，不可用时返回 False。"""
    import db                         # 导入数据库模块
    try:                              # 尝试连接并 ping
        db.get_redis_client().ping()  # 探测连通性
        return True                   # 可用
    except Exception:                 # 连不上
        return False                  # 不可用


def test_short_memory_push_get() -> None:
    """测试短期记忆：push 3 条后 get 返回正序 3 条；Redis 连不上则跳过。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    pytest.importorskip("redis")      # 缺少 redis 依赖时跳过本用例
    if not _redis_ready():            # Redis 不可用
        pytest.skip("Redis 不可用，跳过")   # 跳过本用例
    import memory                     # 导入记忆模块
    user_id = 9901                    # 用一个独立的测试用户编号，避免污染
    memory.clear_short_memory(user_id)   # 先清空
    for i in range(3):                # 依次推入 3 条
        role = "user" if i % 2 == 0 else "assistant"   # 交替角色
        memory.push_short_memory(user_id, role, f"第{i}条")   # 推入一条
    history = memory.get_short_memory(user_id)   # 读回
    assert len(history) == 3, f"应有 3 条，实际 {len(history)}"   # 断言条数
    assert [h["content"] for h in history] == ["第0条", "第1条", "第2条"], "顺序不是正序"   # 断言正序
    assert history[0]["role"] == "user", "角色字段不正确"        # 断言角色
    memory.clear_short_memory(user_id)    # 清理测试数据


def test_short_memory_ltrim() -> None:
    """测试短期记忆上限：推入超过上限的条数，只保留最近 N 条；Redis 连不上则跳过。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    pytest.importorskip("redis")      # 缺少 redis 依赖时跳过本用例
    if not _redis_ready():            # Redis 不可用
        pytest.skip("Redis 不可用，跳过")   # 跳过本用例
    import memory                     # 导入记忆模块
    import config                     # 导入配置模块
    user_id = 9902                    # 独立测试用户
    memory.clear_short_memory(user_id)   # 先清空
    limit = config.LLM_HISTORY_LIMIT  # 上限
    assert limit > 0, "历史上限必须为正数"   # 断言配置合理
    total = limit + 5                 # 多推 5 条
    for i in range(total):            # 依次推入
        memory.push_short_memory(user_id, "user", f"第{i}条")   # 推入一条
    history = memory.get_short_memory(user_id)   # 读回
    assert len(history) == limit, f"应只剩 {limit} 条，实际 {len(history)}"   # 断言被截断
    assert history[-1]["content"] == f"第{total - 1}条", "最新一条没有保留"   # 断言保留最新
    assert history[0]["content"] == f"第{total - limit}条", "最旧一条应被丢弃"   # 断言丢弃最旧
    memory.clear_short_memory(user_id)    # 清理测试数据


def _milvus_ready() -> bool:          # 辅助函数：探测 Milvus 是否可用
    """探测 Milvus 是否可连，不可用时返回 False。"""
    import vector_store               # 导入向量库模块
    try:                              # 尝试列集合
        vector_store.get_milvus_client().list_collections()   # 探测连通性
        return True                   # 可用
    except Exception:                 # 连不上
        return False                  # 不可用


def test_long_memory_roundtrip() -> None:
    """测试长期记忆：存一条后能召回；Milvus 连不上则跳过。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    pytest.importorskip("pymilvus")   # 缺少 pymilvus 依赖时跳过本用例
    if not _milvus_ready():           # Milvus 不可用
        pytest.skip("Milvus 不可用，跳过")   # 跳过本用例
    import memory                     # 导入记忆模块
    user_id = 9901                    # 独立的测试用户编号
    question = "测试问题：变压器油色谱分析多久做一次"     # 待写入的问题
    answer = "测试答案：按标准规定周期执行，具体以现场规程为准。"   # 待写入的答案
    memory.save_long_memory(user_id, question, answer, source="unittest")   # 写入
    hits = memory.search_long_memory(user_id, "色谱分析周期", top_k=3)     # 按相似问题召回
    assert hits, "长期记忆没有召回任何结果"                 # 断言有结果
    joined = " ".join(hit["question"] for hit in hits)     # 拼起召回的问题
    assert question in joined, f"没有召回刚写入的记忆：{joined[:80]}"   # 断言召回目标
    for hit in hits:                  # 逐条检查字段
        for key in ("question", "answer", "score"):        # 三个必备字段
            assert key in hit, f"记忆结果缺少字段：{key}"    # 断言字段齐全


def test_long_memory_user_isolation() -> None:
    """测试长期记忆的用户隔离：两个用户各存一条，交叉检索不串；Milvus 连不上则跳过。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    pytest.importorskip("pymilvus")   # 缺少 pymilvus 依赖时跳过本用例
    if not _milvus_ready():           # Milvus 不可用
        pytest.skip("Milvus 不可用，跳过")   # 跳过本用例
    import memory                     # 导入记忆模块
    q1 = "甲用户的专属问题：分接开关怎么测"      # 用户 1 的问题
    q2 = "乙用户的专属问题：SF6 气体怎么回收"    # 用户 2 的问题
    memory.save_long_memory(9901, q1, "甲用户的答案", source="unittest")   # 写入甲
    memory.save_long_memory(9902, q2, "乙用户的答案", source="unittest")   # 写入乙
    hits1 = memory.search_long_memory(9901, "分接开关 测试", top_k=5)      # 以甲身份检索
    hits2 = memory.search_long_memory(9902, "SF6 回收", top_k=5)           # 以乙身份检索
    text1 = " ".join(hit["question"] for hit in hits1)      # 甲的召回内容
    text2 = " ".join(hit["question"] for hit in hits2)      # 乙的召回内容
    assert q2 not in text1, "甲检索到了乙的记忆，用户隔离失效"   # 断言不串用户
    assert q1 not in text2, "乙检索到了甲的记忆，用户隔离失效"   # 断言不串用户
    assert q1 in text1, "甲没有召回自己的记忆"                  # 断言能召回自己的
    assert q2 in text2, "乙没有召回自己的记忆"                  # 断言能召回自己的


def test_build_messages_history_limit() -> None:
    """测试历史截断：给 20 条历史，只取最近 LLM_HISTORY_LIMIT 条。"""
    import rag                        # 导入问答模块
    import config                     # 导入配置模块
    limit = config.LLM_HISTORY_LIMIT  # 上限
    history = []                      # 构造历史列表
    for i in range(20):               # 生成 20 条
        role = "user" if i % 2 == 0 else "assistant"   # 交替角色
        history.append({"role": role, "content": f"第{i}轮"})   # 追加一条
    messages = rag.build_messages("新问题", [], history)   # 组装
    body = messages[1:-1]             # 去掉 system 与 user 两条
    assert len(body) == limit, f"应只带入 {limit} 条历史，实际 {len(body)}"   # 断言条数
    assert body[-1]["content"] == "第19轮", "没有取最近的历史"   # 断言取的是最近的
    assert body[0]["content"] == f"第{20 - limit}轮", "取历史的起点不对"   # 断言起点正确
    empty = rag.build_messages("新问题", [], [])   # 无历史时组装
    assert len(empty) == 2, "无历史时应只有 system 与 user 两条"   # 断言条数


def test_ask_writes_memory() -> None:
    """测试记忆写回：MySQL 会话消息、Redis 短期记忆都能落库；依赖不可用则跳过。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    if not (_redis_ready() and _milvus_ready()):   # 依赖服务不可用
        pytest.skip("Redis 或 Milvus 不可用，跳过")   # 跳过本用例
    import db_user                    # 导入用户数据模块
    import memory                     # 导入记忆模块
    db_user.init_default_roles()      # 确保预置角色存在
    user_id = 9903                    # 独立的测试用户编号
    if not db_user.get_user(user_id):  # 用户不存在
        created = db_user.register_user(f"unittest_{user_id}", "pwd123456")   # 注册
        if created:                   # 注册成功就用新编号
            user_id = created         # 更新编号
    conversations = db_user.list_conversations(user_id)   # 查已有会话
    if conversations:                 # 已有就复用
        conversation_id = conversations[0]["conversation_id"]   # 取第一个
    else:                             # 没有就新建
        conversation_id = db_user.create_conversation(user_id, 1, "单元测试会话")   # 建一条
    memory.clear_short_memory(user_id)   # 清空短期记忆
    db_user.save_message(conversation_id, "user", "单元测试提问")     # 写一条用户消息
    db_user.save_message(conversation_id, "assistant", "单元测试回答")   # 写一条助手消息
    messages = db_user.list_messages(conversation_id, limit=100)   # 读回消息
    assert len(messages) >= 2, f"消息未落库，只有 {len(messages)} 条"   # 断言落库
    memory.push_short_memory(user_id, "user", "单元测试短期记忆")   # 写短期记忆
    assert memory.get_short_memory(user_id), "短期记忆未写入"       # 断言写入


def test_rewrite_query_no_history() -> None:
    """测试查询改写：history 为空时直接返回原 query，不调模型。"""
    import optimize                   # 导入优化模块
    called = {"n": 0}                 # 计数器，用于确认没有调用模型

    def fake_chat(prompt, max_tokens):   # 伪造的模型调用
        called["n"] += 1               # 计数
        return "不应被调用"             # 返回占位

    original = optimize._chat         # 保存原函数
    optimize._chat = fake_chat        # 替换成伪造版本
    try:                              # 断言完恢复
        assert optimize.rewrite_query("变压器漏油怎么处理", None) == "变压器漏油怎么处理"   # 无历史
        assert optimize.rewrite_query("变压器漏油怎么处理", []) == "变压器漏油怎么处理"     # 空列表
        assert called["n"] == 0, "无历史时不应调用模型"      # 断言未调用
    finally:                          # 恢复原函数
        optimize._chat = original     # 还原


def test_rewrite_query_with_history() -> None:
    """测试查询改写：有历史时调用模型，返回改写后的问题。"""
    import optimize                   # 导入优化模块
    captured = {}                     # 保存收到的提示词

    def fake_chat(prompt, max_tokens):   # 伪造的模型调用
        captured["prompt"] = prompt    # 记下提示词
        return "绝缘油气相色谱分析的具体操作步骤是什么？"   # 返回改写结果

    original = optimize._chat         # 保存原函数
    optimize._chat = fake_chat        # 替换
    try:                              # 断言完恢复
        history = [{"role": "user", "content": "绝缘油气相色谱分析步骤是什么"},
                   {"role": "assistant", "content": "标准规定了仪器准备与取气。"}]   # 构造历史
        result = optimize.rewrite_query("那具体操作步骤呢", history)   # 执行改写
        assert result == "绝缘油气相色谱分析的具体操作步骤是什么？", f"改写结果未生效：{result}"   # 断言结果
        assert "那具体操作步骤呢" in captured["prompt"], "提示词里缺少当前追问"      # 断言含追问
        assert "绝缘油气相色谱分析步骤是什么" in captured["prompt"], "提示词里缺少历史"   # 断言含历史
        assert "用户：" in captured["prompt"] and "助手：" in captured["prompt"], "历史格式不对"   # 断言格式
    finally:                          # 恢复原函数
        optimize._chat = original     # 还原


def test_rewrite_query_llm_failure() -> None:
    """测试查询改写降级：模型调用抛异常时返回原问题，不向上抛错。"""
    import optimize                   # 导入优化模块

    def boom(prompt, max_tokens):     # 必然失败的模型调用
        raise RuntimeError("模拟模型故障")   # 抛出异常

    original = optimize._chat         # 保存原函数
    optimize._chat = boom             # 替换
    try:                              # 断言完恢复
        history = [{"role": "user", "content": "上一轮问题"}]   # 构造历史
        assert optimize.rewrite_query("追问内容", history) == "追问内容"   # 断言降级为原问题
    finally:                          # 恢复原函数
        optimize._chat = original     # 还原


def test_expand_query() -> None:
    """测试查询扩写：返回原问题加若干同义问题；模型失败时只返回原问题。"""
    import optimize                   # 导入优化模块

    def fake_chat(prompt, max_tokens):   # 伪造的模型调用
        return "变压器渗油了该怎么处理\n变压器出现漏油情况应如何解决\n\n变压器漏油了怎么办"   # 含空行

    original = optimize._chat         # 保存原函数
    optimize._chat = fake_chat        # 替换
    try:                              # 断言完恢复
        results = optimize.expand_query("变压器漏油怎么处理", 3)   # 执行扩写
        assert results[0] == "变压器漏油怎么处理", "原问题应排在第一位"   # 断言原问题在首位
        assert len(results) == 4, f"应为原问题 + 3 条同义问题，实际 {len(results)}"   # 断言条数
        assert "" not in results, "扩写结果里混入了空行"          # 断言空行被过滤

        def boom(prompt, max_tokens):  # 必然失败的调用
            raise RuntimeError("模拟模型故障")   # 抛异常
        optimize._chat = boom          # 替换
        assert optimize.expand_query("变压器漏油怎么处理", 3) == ["变压器漏油怎么处理"]   # 断言降级
    finally:                          # 恢复原函数
        optimize._chat = original     # 还原


def test_cache_roundtrip() -> None:
    """测试缓存读写：写入后能读出相同内容；Redis 连不上则跳过。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    pytest.importorskip("redis")      # 缺少 redis 依赖时跳过本用例
    if not _redis_ready():            # Redis 不可用
        pytest.skip("Redis 不可用，跳过")   # 跳过本用例
    import optimize                   # 导入优化模块
    key = "unittest_roundtrip"        # 测试用键
    payload = {"answer": "测试答案", "sources": [{"source": "x.pdf", "score": 0.9}]}   # 测试用值
    optimize.cache_set(key, payload, 60)      # 写入
    got = optimize.cache_get(key)             # 读出
    assert got == payload, f"缓存内容不一致：{got}"          # 断言内容一致
    assert optimize.cache_get("unittest_not_exist") is None, "不存在的键应返回 None"   # 断言未命中


def test_generate_summary() -> None:
    """测试摘要生成：长度不超过 max_len，且非空。"""
    import enrich                     # 导入增强模块
    text = ("变压器漏油是电力设备常见缺陷。发现漏油后应先断电并验电。"
            "变压器漏油的常见原因是密封胶垫老化。处理变压器漏油需要更换密封件。")   # 测试文本
    summary = enrich.generate_summary(text, 120)       # 生成摘要
    assert summary, "摘要为空"                          # 断言非空
    assert len(summary) <= 120, f"摘要超长：{len(summary)}"   # 断言长度上限
    short = enrich.generate_summary("变压器漏油。", 120)   # 短文本
    assert short == "变压器漏油。", "短文本应原样返回"      # 断言短文本处理
    assert enrich.generate_summary("", 120) == "", "空文本应返回空串"   # 断言空文本
    tiny = enrich.generate_summary(text, 20)           # 很短的 max_len
    assert len(tiny) <= 20, f"极短上限下仍超长：{len(tiny)}"   # 断言严格截断


def test_remove_low_quality() -> None:
    """测试删除低质块：过短、过长、空白占比高、数字占比高都应被丢弃。"""
    import enrich                     # 导入增强模块
    chunks = [                        # 构造各类块
        {"text": "太短"},                                        # 过短
        {"text": "正" * 5000},                                   # 过长
        {"text": " " * 200},                                     # 空白占比高（同时过短）
        {"text": "1234567890" * 30},                             # 数字占比高（300 字）
        {"text": "正常的电力标准条款内容。" * 10},                 # 正常，保留
    ]
    kept, dropped = enrich.remove_low_quality(chunks, 80, 2000)   # 执行过滤
    assert len(kept) == 1, f"应只保留 1 条，实际 {len(kept)}"        # 断言保留数
    assert dropped == 4, f"应丢弃 4 条，实际 {dropped}"              # 断言丢弃数
    assert kept[0]["text"].startswith("正常的电力标准条款内容"), "保留的不是正常块"   # 断言保留内容
    all_ok = [{"text": "正常内容" * 30}]                            # 全部合格的输入
    kept2, dropped2 = enrich.remove_low_quality(all_ok, 80, 2000)   # 再跑一次
    assert len(kept2) == 1 and dropped2 == 0, "合格块被误删"         # 断言不误删


def test_remove_low_quality_boundaries() -> None:
    """测试删除低质块的边界：恰好等于长度上下限的块应保留。"""
    import enrich                     # 导入增强模块
    just_min = {"text": "边" * 80}    # 恰好等于下限
    just_max = {"text": "界" * 2000}  # 恰好等于上限
    kept, dropped = enrich.remove_low_quality([just_min, just_max], 80, 2000)   # 执行过滤
    assert len(kept) == 2, f"边界值应保留，实际保留 {len(kept)}"    # 断言边界保留
    assert dropped == 0, f"边界值不应被丢弃，实际丢弃 {dropped}"      # 断言未丢弃


def test_dedup_chunks() -> None:
    """测试语义去重：近似文本只留一条，且优先保留较短的那条。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    import enrich                     # 导入增强模块
    import vector_store               # 导入向量库模块
    try:                              # 需要 BGE-M3 才能算相似度
        vector_store.load_bge_m3()    # 提前加载，失败就走 skip
    except Exception as exc:          # 模型不可用
        pytest.skip(f"BGE-M3 不可用，跳过：{exc}")   # 跳过本用例
    text = "变压器漏油应先断电，再检查油箱焊缝与密封胶垫，按力矩要求复紧法兰螺栓。"   # 基准文本
    chunks = [                        # 构造三条块：两条几乎相同，一条完全不同
        {"text": text},               # 基准
        {"text": text + "同时应记录处理过程。"},   # 与基准高度相似
        {"text": "六氟化硫气体回收应采用专用装置，禁止直接排入大气。"},   # 完全不同
    ]
    kept, dropped = enrich.dedup_chunks(chunks, 0.95)     # 执行去重
    assert dropped == 1, f"应丢弃 1 条重复，实际 {dropped}"     # 断言丢弃数
    assert len(kept) == 2, f"应保留 2 条，实际 {len(kept)}"      # 断言保留数
    texts = [c["text"] for c in kept]                   # 取出保留文本
    assert text in texts, "较短的那条应被优先保留"        # 断言短的优先保留
    assert any("六氟化硫" in t for t in texts), "不同的内容被误删"   # 断言不同内容保留
    kept_tight, dropped_tight = enrich.dedup_chunks(chunks, 0.99)   # 阈值提高
    assert dropped_tight == 0, "阈值提高到 0.99 后不应再判为重复"    # 断言阈值生效


def test_build_parent_child() -> None:
    """测试父子块：6 个子块、parent_size=3，应得 2 个父块、6 个子块、parent_id 正确。"""
    import enrich                     # 导入增强模块
    chunks = [{"text": f"子{i}", "source": "x.pdf", "page": i} for i in range(6)]   # 6 个子块
    result = enrich.build_parent_child(chunks, 3)      # 构建父子块
    parents = [c for c in result if c.get("is_parent")]        # 取出父块
    children = [c for c in result if not c.get("is_parent")]   # 取出子块
    assert len(parents) == 2, f"应有 2 个父块，实际 {len(parents)}"      # 断言父块数
    assert len(children) == 6, f"应有 6 个子块，实际 {len(children)}"    # 断言子块数
    assert len(result) == 8, f"总记录应为 8 条，实际 {len(result)}"      # 断言总数
    assert parents[0]["text"] == "子0子1子2", f"父块文本拼接不对：{parents[0]['text']}"   # 断言拼接
    assert parents[1]["text"] == "子3子4子5", f"父块文本拼接不对：{parents[1]['text']}"   # 断言拼接
    assert parents[0]["child_ids"] == [1, 2, 3], f"child_ids 不对：{parents[0]['child_ids']}"   # 断言子块编号
    assert parents[1]["child_ids"] == [5, 6, 7], f"child_ids 不对：{parents[1]['child_ids']}"   # 断言子块编号
    assert [c["parent_id"] for c in children] == [0, 0, 0, 4, 4, 4], "子块的 parent_id 不对"   # 断言归属
    assert all(c["parent_id"] == -1 for c in parents), "父块自身的 parent_id 应为 -1"   # 断言父块标记
    assert result[0]["is_parent"] and result[4]["is_parent"], "父块应插在各自那组子块之前"   # 断言位置


def test_get_system_prompt_by_role() -> None:
    """测试按角色切换提示词：两个角色返回不同提示词，未知角色回退专家版。"""
    import prompt                     # 导入提示词模块
    expert = prompt.get_system_prompt("power_repair_expert")     # 专家角色
    general = prompt.get_system_prompt("general_assistant")      # 通用助手角色
    assert expert == prompt.SYSTEM_PROMPT, "专家角色没有返回专家提示词"       # 断言映射
    assert general == prompt.SYSTEM_PROMPT_GENERAL, "通用助手没有返回简短提示词"   # 断言映射
    assert expert != general, "两个角色的提示词相同，切换未生效"           # 断言不同
    assert len(general) < len(expert), "通用助手提示词应比专家版简短"       # 断言简短
    for unknown in ("", None, "no_such_role"):                      # 未知角色
        assert prompt.get_system_prompt(unknown) == prompt.SYSTEM_PROMPT, \
            f"未知角色 {unknown!r} 未回退到专家提示词"                   # 断言回退
    import rag                        # 导入问答模块
    ctx = [{"source": "x.pdf", "text": "条款", "score": 0.9}]        # 构造一条上下文
    m_expert = rag.build_messages("q", ctx, [], "power_repair_expert")   # 专家角色的消息
    m_general = rag.build_messages("q", ctx, [], "general_assistant")    # 通用助手的消息
    assert m_expert[0]["content"] != m_general[0]["content"], "messages[0] 未按角色切换"   # 断言切换
    assert m_general[0]["content"] == prompt.SYSTEM_PROMPT_GENERAL, "messages[0] 内容不对"   # 断言内容


def _api_client():                    # 辅助函数：创建 HTTP 测试客户端
    """创建 FastAPI 测试客户端；缺少依赖时返回 None。"""
    try:                              # 导入失败说明环境缺包
        from fastapi.testclient import TestClient   # 导入测试客户端
        import app as app_module      # 导入应用模块
        return TestClient(app_module.app)           # 返回客户端
    except Exception:                 # 导入失败
        return None                   # 返回 None


def test_api_health() -> None:
    """测试健康检查接口：/health 返回 200 且带三项依赖状态。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    client = _api_client()            # 创建测试客户端
    if client is None:                # 客户端创建失败
        pytest.skip("FastAPI TestClient 不可用，跳过")   # 跳过本用例
    resp = client.get("/health")      # 请求健康检查
    assert resp.status_code == 200, f"/health 状态码异常：{resp.status_code}"   # 断言状态码
    body = resp.json()                # 解析响应
    assert body.get("status") == "ok", f"status 字段异常：{body}"      # 断言服务状态
    for key in ("milvus", "mysql", "redis"):        # 三项依赖状态
        assert key in body, f"缺少依赖状态字段：{key}"                 # 断言字段存在
        assert isinstance(body[key], bool), f"{key} 不是布尔值"        # 断言类型


def test_api_register_login() -> None:
    """测试注册与登录接口：注册成功返回 200，登录返回非空 token；MySQL 不可用则跳过。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    if not _mysql_ready():            # MySQL 不可用
        pytest.skip("MySQL 不可用，跳过")   # 跳过本用例
    client = _api_client()            # 创建测试客户端
    if client is None:                # 客户端创建失败
        pytest.skip("FastAPI TestClient 不可用，跳过")   # 跳过本用例
    import time                       # 导入时间模块，用于生成唯一用户名
    username = f"apitest_{int(time.time() * 1000)}"   # 用毫秒时间戳保证用户名唯一
    resp = client.post("/auth/register", json={"username": username, "password": "pwd123456"})   # 注册
    assert resp.status_code == 200, f"注册失败：{resp.status_code} {resp.text}"   # 断言状态码
    body = resp.json()                # 解析响应
    assert body.get("user_id"), f"注册未返回用户编号：{body}"        # 断言返回编号
    duplicate = client.post("/auth/register", json={"username": username, "password": "pwd123456"})   # 重复注册
    assert duplicate.status_code == 400, f"重复注册应返回 400，实际 {duplicate.status_code}"   # 断言 400
    login = client.post("/auth/login", json={"username": username, "password": "pwd123456"})   # 登录
    assert login.status_code == 200, f"登录失败：{login.status_code} {login.text}"   # 断言状态码
    token = login.json().get("token")   # 取出令牌
    assert token, "登录未返回 token"      # 断言令牌非空
    assert len(token) == 64, f"token 长度异常：{len(token)}"        # 断言长度
    bad = client.post("/auth/login", json={"username": username, "password": "wrong"})   # 错误密码
    assert bad.status_code == 401, f"错误密码应返回 401，实际 {bad.status_code}"   # 断言 401


def test_api_roles() -> None:
    """测试角色接口：/roles 返回至少两条角色。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    if not _mysql_ready():            # MySQL 不可用
        pytest.skip("MySQL 不可用，跳过")   # 跳过本用例
    client = _api_client()            # 创建测试客户端
    if client is None:                # 客户端创建失败
        pytest.skip("FastAPI TestClient 不可用，跳过")   # 跳过本用例
    resp = client.get("/roles")       # 请求角色列表
    assert resp.status_code == 200, f"/roles 状态码异常：{resp.status_code}"   # 断言状态码
    roles = resp.json().get("roles", [])   # 取出角色列表
    assert len(roles) >= 2, f"角色应至少 2 条，实际 {len(roles)}"      # 断言条数
    names = {r["role_name"] for r in roles}                          # 角色名集合
    assert "power_repair_expert" in names, "缺少电力维修专家角色"      # 断言预置角色
    assert "general_assistant" in names, "缺少通用助手角色"            # 断言预置角色


def test_api_chat_nonstream(monkeypatch) -> None:
    """测试问答接口（非流式）：mock 掉 rag.ask，断言返回结构与 ChatResp 一致。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    client = _api_client()            # 创建测试客户端
    if client is None:                # 客户端创建失败
        pytest.skip("FastAPI TestClient 不可用，跳过")   # 跳过本用例
    import app as app_module          # 导入应用模块

    def fake_ask(query, user_id, role_id, conversation_id, top_k=None, stream=False):   # 伪造问答
        return {"answer": "测试答案", "sources": [{"source": "x.pdf", "summary": "片段", "score": 0.9}],
                "rewritten_query": query, "cache_hit": True, "conversation_id": conversation_id,
                "usage": {"total_tokens": 12}, "contexts": [], "raw": "", "check": {},
                "history_used": 0, "long_memory_used": 0}   # 返回结构完整的假结果

    monkeypatch.setattr(app_module.rag, "ask", fake_ask)   # 替换真实问答
    resp = client.post("/chat", json={"user_id": 1, "role_id": 1, "conversation_id": 1,
                                      "query": "测试问题", "stream": False})   # 请求问答
    assert resp.status_code == 200, f"/chat 状态码异常：{resp.status_code} {resp.text}"   # 断言状态码
    body = resp.json()                # 解析响应
    for key in ("answer", "sources", "rewritten_query", "cache_hit", "conversation_id", "usage"):
        assert key in body, f"响应缺少字段：{key}"       # 断言字段齐全
    assert body["answer"] == "测试答案", f"答案不正确：{body['answer']}"      # 断言答案
    assert body["cache_hit"] is True, "cache_hit 未透传"                    # 断言缓存标记
    assert body["conversation_id"] == 1, "conversation_id 未透传"           # 断言会话编号


def test_rrf_weight_mysql() -> None:
    """测试 MySQL 路降权：降权后同名词条排在知识路之后。"""
    import retrieval                  # 导入检索模块
    import config                     # 导入配置模块
    milvus = [{"id": i + 1, "text": f"M{i}", "rank": i} for i in range(3)]   # 知识路候选
    mysql = [{"id": i + 1, "text": f"S{i}", "rank": i} for i in range(3)]    # 历史对话路候选
    heavy = retrieval.fuse_multi_route([(milvus, 1.0, "milvus"), (mysql, 1.0, "mysql")])   # 等权融合
    light = retrieval.fuse_multi_route(
        [(milvus, 1.0, "milvus"), (mysql, config.MYSQL_RECALL_WEIGHT, "mysql")])   # 降权融合
    scores_heavy = {item["text"]: item["score"] for item in heavy}   # 等权下的分数
    scores_light = {item["text"]: item["score"] for item in light}   # 降权下的分数
    assert abs(scores_heavy["M0"] - scores_heavy["S0"]) < 1e-9, "等权时两路首条应同分"   # 断言等权
    assert scores_light["S0"] < scores_light["M0"], "降权后 MySQL 首条未低于知识路首条"   # 断言降权
    assert scores_light["S0"] < scores_light["M1"], "降权后 MySQL 首条仍压过知识路第二条"   # 断言幅度
    ratio = scores_light["S0"] / scores_light["M0"]                  # 实际权重比
    # score 字段被四舍五入到 6 位小数，比值会有约 1e-5 量级的舍入误差，因此容差取 1e-3
    assert abs(ratio - config.MYSQL_RECALL_WEIGHT) < 1e-3, f"权重比不是配置值：{ratio}"   # 断言比例
    for key in ("M0", "M1", "M2"):    # 知识路的三条
        assert key in scores_light, f"融合后丢失知识路候选：{key}"     # 断言未丢失


def test_parent_child_lookup() -> None:
    """测试父子块回溯：命中子块的结果应带上 parent_text。"""
    import retrieval                  # 导入检索模块
    cache = retrieval._load_parent_cache()   # 加载父块缓存
    if not cache:                     # 没有父块数据
        pytest.skip("父块文件不存在，跳过")   # 跳过本用例
    parent_id = next(iter(cache))     # 取一个存在的父块索引
    results = [{"text": "子块甲", "parent_id": parent_id},        # 有父块的子块
               {"text": "子块乙", "parent_id": -1},               # 没有父块的子块
               {"text": "子块丙", "parent_id": 10 ** 9}]          # 父块索引不存在的子块
    out = retrieval.attach_parent_text([dict(item) for item in results])   # 附加父块
    assert out[0].get("extra", {}).get("parent_text") == cache[parent_id], "父块文本未附加或内容不符"   # 断言附件
    assert "extra" not in out[1] or "parent_text" not in out[1]["extra"], "无父块的结果不应有 parent_text"   # 断言无父块
    assert "extra" not in out[2] or "parent_text" not in out[2]["extra"], "父块索引无效时不应有内容"   # 断言越界保护
    import rag                        # 导入问答模块
    messages = rag.build_messages("测试问题", out, [], "power_repair_expert")   # 组装消息
    user_text = messages[-1]["content"]   # 取出用户消息
    assert "命中片段：子块甲" in user_text, "上下文里没有命中片段"        # 断言格式
    assert "上下文扩充：" in user_text, "上下文里没有上下文扩充"          # 断言父块已拼入
    assert user_text.count("上下文扩充：") == 1, "父块数量不对"           # 断言只拼了一条


def test_normalize_query() -> None:
    """测试查询归一化：不同标点、空格、大小写归一化后结果相同。"""
    import optimize                   # 导入优化模块
    base = optimize.normalize_query("Transformer漏油怎么处理")   # 基准
    for variant in ("Transformer漏油怎么处理？",     # 中文问号
                    "Transformer漏油，怎么处理！",    # 中文逗号与叹号
                    "  Transformer漏油怎么处理。  ",   # 首尾空格与句号
                    "Transformer漏油怎么处理.",       # 英文句点
                    "Transformer漏油怎么处理!?"):     # 英文叹号问号
        assert optimize.normalize_query(variant) == base, f"归一化结果不一致：{variant!r}"   # 断言一致
    assert base == "transformer漏油怎么处理", f"归一化结果不符预期：{base}"   # 断言小写化
    assert optimize.normalize_query("变压器  漏油") == "变压器 漏油", "多个空格未合并"   # 断言空格合并
    assert optimize.normalize_query("") == "", "空串应返回空串"      # 断言空输入
    assert optimize.normalize_query(None) == "", "None 应返回空串"   # 断言 None 输入
    assert optimize.normalize_query("（变压器）【漏油】") == "变压器漏油", "括号未去掉"   # 断言括号


def test_cache_key_stable() -> None:
    """测试缓存键稳定性：同一问题不同标点问法生成同一个键。"""
    import optimize                   # 导入优化模块
    keys = {optimize.cache_key(7, "power_repair_expert", q) for q in
            ("变压器漏油怎么处理", "变压器漏油怎么处理？", "变压器漏油，怎么处理！",
             " 变压器漏油怎么处理。 ")}       # 四种问法的键
    assert len(keys) == 1, f"同一问题的键不唯一：{keys}"       # 断言唯一
    key = keys.pop()                  # 取出该键
    assert key.startswith("cache:rag:answer:7:power_repair_expert:"), f"键结构不对：{key}"   # 断言结构
    other_user = optimize.cache_key(8, "power_repair_expert", "变压器漏油怎么处理")   # 换用户
    assert other_user != key, "不同用户的键相同，缓存会串用户"     # 断言用户隔离
    other_role = optimize.cache_key(7, "general_assistant", "变压器漏油怎么处理")     # 换角色
    assert other_role != key, "不同角色的键相同，缓存会串角色"     # 断言角色隔离


def test_long_memory_dedup() -> None:
    """测试长期记忆去重：相同 question 只保留分数最高的一条。"""
    import rag                        # 导入问答模块
    memories = [                      # 构造 3 条相同问题、分数不同的记忆
        {"question": "变压器漏油怎么处理", "answer": "答案A", "score": 0.5},
        {"question": "变压器漏油怎么处理", "answer": "答案B", "score": 0.9},
        {"question": "变压器漏油怎么处理", "answer": "答案C", "score": 0.7},
        {"question": "绝缘油怎么测", "answer": "答案D", "score": 0.6},
    ]
    original = rag.memory.search_long_memory     # 保存原函数
    rag.memory.search_long_memory = lambda uid, q, k=None: memories   # 替换为固定返回
    try:                              # 断言完恢复
        result = rag.collect_long_memory("任意问题", 1)   # 执行去重
        assert len(result) == 2, f"应去重为 2 条，实际 {len(result)}"     # 断言条数
        top = [m for m in result if m["question"] == "变压器漏油怎么处理"][0]   # 取该问题的记忆
        assert top["answer"] == "答案B", f"没有保留分数最高的那条：{top['answer']}"   # 断言保留最高分
        assert top["score"] == 0.9, f"分数不对：{top['score']}"           # 断言分数
    finally:                          # 恢复原函数
        rag.memory.search_long_memory = original


def test_long_memory_not_in_rerank(monkeypatch) -> None:
    """测试长期记忆不进精排：rerank 收到的候选里没有长期记忆，但提示词里有。"""
    import rag                        # 导入问答模块
    captured = {}                     # 记录各阶段入参
    retrieved = [{"id": 1, "text": "标准条款原文", "source": "GBT 1.pdf",
                  "summary": "", "score": 0.9}]     # 伪造的检索结果
    memories = [{"question": "历史问题", "answer": "历史回答", "score": 0.99}]   # 伪造的长期记忆

    def fake_rerank(query, docs, top_k=None):       # 伪造精排，记录入参
        captured["rerank_docs"] = list(docs)        # 记下送进精排的候选
        return docs                                 # 原样返回

    def fake_ask_llm(messages):                     # 伪造大模型调用，记录提示词
        captured["messages"] = messages             # 记下消息
        return {"content": "测试答案", "usage": {"total_tokens": 1}}   # 返回假结果

    monkeypatch.setattr(rag.retrieval, "rerank", fake_rerank)            # 替换精排（记录入参）
    # 只替换召回阶段，保留 retrieve 本体，这样它会真正调用被替换的 rerank
    monkeypatch.setattr(rag.retrieval, "multi_recall", lambda q, k=None: list(retrieved))
    monkeypatch.setattr(rag.memory, "search_long_memory", lambda uid, q, k=None: memories)   # 替换记忆
    monkeypatch.setattr(rag.memory, "get_short_memory", lambda uid: [])   # 无短期记忆
    monkeypatch.setattr(rag, "call_deepseek", fake_ask_llm)               # 替换大模型
    monkeypatch.setattr(rag, "save_turn", lambda *a, **k: None)           # 跳过写回
    monkeypatch.setattr(rag.optimize, "cache_get", lambda key: None)      # 不命中缓存
    monkeypatch.setattr(rag.optimize, "cache_set", lambda key, value, expire=None: None)   # 跳过写缓存
    monkeypatch.setattr(rag, "_role_name_of", lambda role_id: "power_repair_expert")   # 固定角色名
    monkeypatch.setattr(rag.optimize, "rewrite_query", lambda q, h=None: q)   # 不做改写

    result = rag.ask("测试问题", 1, 1, 1, stream=False)     # 执行问答

    docs = captured["rerank_docs"]                   # 精排收到的候选
    assert len(docs) == 1, f"送进精排的候选数不对：{len(docs)}"          # 断言只有检索结果
    for doc in docs:                                 # 逐条检查
        assert doc.get("source") != "历史对话记忆", "长期记忆进入了精排"   # 断言没有记忆
    user_msg = captured["messages"][-1]["content"]   # 取出用户消息
    assert "历史对话记忆" in user_msg, "长期记忆没有拼进提示词"          # 断言提示词里有
    assert "优先级低于标准条款" in user_msg, "提示词没有写明优先级"      # 断言优先级声明
    assert "标准条款原文" in user_msg, "标准条款没有进提示词"            # 断言条款在
    assert result["long_memory_used"] == 1, f"long_memory_used 不对：{result['long_memory_used']}"   # 断言计数


def test_cache_clear_by_user() -> None:
    """测试 /cache/clear 按用户清理：只清目标用户的缓存键；Redis 不可用则跳过。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    pytest.importorskip("redis")      # 缺少 redis 依赖时跳过本用例
    if not _redis_ready():            # Redis 不可用
        pytest.skip("Redis 不可用，跳过")   # 跳过本用例
    client = _api_client()            # 创建测试客户端
    if client is None:                # 客户端创建失败
        pytest.skip("FastAPI TestClient 不可用，跳过")   # 跳过本用例
    import optimize                   # 导入优化模块
    import db                         # 导入数据库模块
    redis_client = db.get_redis_client()   # 取 Redis 客户端
    key_a = optimize.cache_key(8801, "power_repair_expert", "单元测试问题甲")   # 用户 A 的键
    key_b = optimize.cache_key(8802, "power_repair_expert", "单元测试问题乙")   # 用户 B 的键
    optimize.cache_set(key_a, {"answer": "甲"}, 60)   # 写入 A 的缓存
    optimize.cache_set(key_b, {"answer": "乙"}, 60)   # 写入 B 的缓存
    assert redis_client.exists(key_a) and redis_client.exists(key_b), "测试缓存未写入"   # 断言前置条件
    resp = client.get("/cache/clear", params={"user_id": 8801})   # 清理用户 A
    assert resp.status_code == 200, f"/cache/clear 状态码异常：{resp.status_code}"   # 断言状态码
    body = resp.json()                # 解析响应
    assert body.get("cache_keys", 0) >= 1, f"未清理到缓存键：{body}"        # 断言清到键
    assert "short_memory" in body, f"响应缺少 short_memory 字段：{body}"    # 断言字段存在
    assert not redis_client.exists(key_a), "用户 A 的缓存未被清理"          # 断言 A 已清
    assert redis_client.exists(key_b), "用户 B 的缓存被误删"                # 断言 B 保留
    redis_client.delete(key_b)        # 清理测试数据


def test_lifespan_migration() -> None:
    """测试 lifespan 迁移：应用已挂 lifespan 上下文，且不再有 on_event 启动钩子。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    import app as app_module          # 导入应用模块
    assert hasattr(app_module, "lifespan"), "未定义 lifespan 函数"     # 断言函数存在
    assert getattr(app_module.app.router, "lifespan_context", None) is not None, \
        "FastAPI 未挂载 lifespan_context"                            # 断言已挂载
    import inspect                    # 导入反射工具
    assert not hasattr(app_module, "on_startup"), "旧的 on_startup 未删除"            # 断言旧钩子已删
    src = inspect.getsource(app_module)                  # 取源码
    assert "on_event" not in src, "源码里仍残留 on_event"   # 断言无 on_event


def test_save_turn_writes_both() -> None:
    """测试 save_turn 写入完整：MySQL 与 Redis 都必须写 user + assistant 两条。

    第 6 步的 bug 是把两次写入用 `or` 串在一行，save_message 返回 lastrowid（真值），
    于是 assistant 那条被短路吞掉。本用例正面卡住这个回归。
    """
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    if not _mysql_ready():            # MySQL 不可用时无法核对落库
        pytest.skip("MySQL 不可用，跳过")   # 跳过本用例
    if not _redis_ready():            # Redis 不可用时无法核对短期记忆
        pytest.skip("Redis 不可用，跳过")   # 跳过本用例
    import json                       # 导入 json，用于解析短期记忆的 JSON 元素
    import db                         # 导入数据库模块
    import db_user                    # 导入用户数据模块
    import rag                        # 导入问答模块
    user_id = 8790                    # 专用测试用户编号，避开真实数据
    redis_client = db.get_redis_client()            # 取 Redis 客户端
    redis_client.delete(f"mem:short:{user_id}")     # 先清空，保证条数确定
    conversation_id = db_user.create_conversation(user_id, 1, "单元测试 save_turn")   # 建测试会话
    rag.save_turn(user_id, conversation_id, "单元测试提问", "单元测试回答")   # 执行本轮写入
    conn = db.get_mysql_conn()        # 取 MySQL 连接
    try:                              # 收尾关连接
        with conn.cursor() as cursor:  # 打开游标
            cursor.execute("SELECT role, content FROM messages WHERE conversation_id = %s",
                           (conversation_id,))   # 只查本会话
            rows = cursor.fetchall()   # 取全部行
    finally:                          # 收尾
        conn.close()                   # 关闭连接
    assert len(rows) == 2, f"MySQL 应写 2 条，实际 {len(rows)}（assistant 很可能被 or 短路吞掉）"
    assert {r["role"] for r in rows} == {"user", "assistant"}, f"MySQL 角色不齐：{rows}"   # 断言角色
    assistant = [r for r in rows if r["role"] == "assistant"][0]   # 取出助手那条
    assert assistant["content"] == "单元测试回答", f"助手内容不对：{assistant['content']}"   # 断言内容
    items = redis_client.lrange(f"mem:short:{user_id}", 0, -1)     # 取短期记忆（最新的在前）
    assert len(items) >= 2, f"Redis 应至少写 2 条，实际 {len(items)}"   # 断言条数
    recent_roles = {json.loads(i)["role"] for i in items[:2]}      # 最近两条的角色
    assert recent_roles == {"user", "assistant"}, f"Redis 最近两条角色不齐：{recent_roles}"   # 断言角色
    redis_client.delete(f"mem:short:{user_id}")     # 清理 Redis 测试数据
    conn = db.get_mysql_conn()        # 再取一次连接，清掉本用例造的会话与消息
    try:                              # 收尾关连接
        with conn.cursor() as cursor:  # 打开游标
            cursor.execute("DELETE FROM messages WHERE conversation_id = %s", (conversation_id,))   # 删消息
            cursor.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))           # 删会话
        conn.commit()                  # 提交删除
    finally:                          # 收尾
        conn.close()                   # 关闭连接


def test_cache_stats_api() -> None:
    """测试 /cache/stats：造两个用户的缓存键，断言统计能覆盖到它们。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    pytest.importorskip("redis")      # 缺少 redis 依赖时跳过本用例
    if not _redis_ready():            # Redis 不可用
        pytest.skip("Redis 不可用，跳过")   # 跳过本用例
    client = _api_client()            # 创建测试客户端
    if client is None:                # 客户端创建失败
        pytest.skip("FastAPI TestClient 不可用，跳过")   # 跳过本用例
    import optimize                   # 导入优化模块，复用缓存键规则
    import db                         # 导入数据库模块
    redis_client = db.get_redis_client()   # 取 Redis 客户端
    key_a = optimize.cache_key(8801, "power_repair_expert", "缓存统计问题甲")   # 用户 8801 的键
    key_b = optimize.cache_key(8802, "power_repair_expert", "缓存统计问题乙")   # 用户 8802 的键
    optimize.cache_set(key_a, {"answer": "甲"}, 60)   # 写入甲的缓存
    optimize.cache_set(key_b, {"answer": "乙"}, 60)   # 写入乙的缓存
    resp = client.get("/cache/stats")   # 请求缓存统计
    assert resp.status_code == 200, f"/cache/stats 状态码异常：{resp.status_code}"   # 断言状态码
    body = resp.json()                # 解析响应
    assert body.get("total_keys", 0) >= 2, f"total_keys 应不少于 2：{body}"     # 断言键总数
    assert body["by_user"].get("8801", 0) >= 1, f"by_user 缺 8801：{body['by_user']}"   # 断言分组甲
    assert body["by_user"].get("8802", 0) >= 1, f"by_user 缺 8802：{body['by_user']}"   # 断言分组乙
    assert "short_memory_users" in body, f"响应缺少 short_memory_users：{body}"   # 断言字段存在
    redis_client.delete(key_a, key_b)   # 清理测试数据


def _conv_messages(conv_id: int) -> list:   # 辅助函数：查某会话的消息行
    """返回某会话的消息行（只取 role 够用），每次都用新连接，避免事务快照干扰。"""
    import db                              # 导入数据库模块
    conn = db.get_mysql_conn()             # 取连接
    try:                                   # 收尾关连接
        with conn.cursor() as cur:         # 打开游标
            cur.execute("SELECT role FROM messages WHERE conversation_id = %s", (conv_id,))
            return list(cur.fetchall())    # 转成 list，方便直接和 [] 比较
    finally:                               # 收尾
        conn.close()                       # 关闭连接


def _drop_conversations(conv_ids: list) -> None:   # 辅助函数：清掉测试造的会话与消息
    """删除给定会话及其消息，供用例收尾使用。"""
    import db                              # 导入数据库模块
    conn = db.get_mysql_conn()             # 取连接
    try:                                   # 收尾关连接
        with conn.cursor() as cur:         # 打开游标
            for cid in conv_ids:           # 逐个会话清理
                cur.execute("DELETE FROM messages WHERE conversation_id = %s", (cid,))    # 删消息
                cur.execute("DELETE FROM conversations WHERE id = %s", (cid,))            # 删会话
        conn.commit()                      # 提交
    finally:                               # 收尾
        conn.close()                       # 关闭连接


def test_delete_conversation_cascade() -> None:
    """测试删除会话级联删消息：删完该会话 messages 应为 0，且越权删除不动别人的数据。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    if not _mysql_ready():            # MySQL 不可用
        pytest.skip("MySQL 不可用，跳过")   # 跳过本用例
    import db_user                    # 导入用户数据模块
    user_id = 8796                    # 专用测试用户编号，避开真实数据
    conv_id = db_user.create_conversation(user_id, 1, "单元测试 级联删除")   # 建测试会话
    db_user.save_message(conv_id, "user", "级联测试提问")        # 写一条用户提问
    db_user.save_message(conv_id, "assistant", "级联测试回答")   # 写一条助手回答
    assert len(_conv_messages(conv_id)) == 2, "前置条件不成立：消息没写进去"   # 断言前置条件
    affected = db_user.delete_conversation(conv_id, user_id)     # 删除该会话
    assert affected == 1, f"应删掉 1 个会话，实际 {affected}"      # 断言返回的会话条数
    left = _conv_messages(conv_id)                               # 再查该会话的消息
    assert left == [], f"级联删除未生效，还剩 {len(left)} 条消息"   # 断言消息已清空
    conv2 = db_user.create_conversation(user_id, 1, "单元测试 越权删除")   # 再建一个会话
    db_user.save_message(conv2, "user", "越权测试提问")                  # 写一条提问
    assert db_user.delete_conversation(conv2, 99999) == 0, "越权删除应返回 0"   # 断言返回 0
    assert len(_conv_messages(conv2)) == 1, "越权删除把别人的消息删掉了"        # 断言数据还在
    _drop_conversations([conv_id, conv2])                        # 清理测试数据


def test_cleanup_orphan_conversations() -> None:
    """测试一次性清理工具：能清掉半截会话，且预期清单不符时会中止不删。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    if not _mysql_ready():            # MySQL 不可用
        pytest.skip("MySQL 不可用，跳过")   # 跳过本用例
    import db_user                    # 导入用户数据模块
    user_id = 8797                    # 专用测试用户编号
    conv_id = db_user.create_conversation(user_id, 1, "单元测试 半截会话")   # 建测试会话
    db_user.save_message(conv_id, "user", "只有提问、没有回答")             # 只写提问，造半截
    assert len(_conv_messages(conv_id)) == 1, "前置条件不成立"              # 断言前置条件
    # 传预期清单 = 只应有这一个半截会话；若库里还有别的，保护机制会中止并让本用例失败
    result = db_user.cleanup_orphan_conversations(expected_ids=[conv_id])   # 执行清理
    assert result["conversations"] == [conv_id], f"该半截会话未被清理：{result}"   # 断言被清掉
    assert result["messages"] >= 1, f"应至少删除 1 条消息：{result}"            # 断言删了消息
    assert _conv_messages(conv_id) == [], "半截会话的消息没被清掉"               # 断言消息清空
    # 保护机制：预期清单与实际不符时应中止，一条都不删
    conv2 = db_user.create_conversation(user_id, 1, "单元测试 保护机制")   # 再造一个半截会话
    db_user.save_message(conv2, "user", "半截提问")                       # 只写提问
    blocked = db_user.cleanup_orphan_conversations(expected_ids=[123456])   # 故意传错清单
    assert blocked == {"conversations": [], "messages": 0}, f"保护机制失效：{blocked}"   # 断言中止
    assert len(_conv_messages(conv2)) == 1, "保护机制失效，数据被误删"       # 断言数据还在
    _drop_conversations([conv_id, conv2])                                 # 清理测试数据


def _clear_user_cache(redis_client, user_id: int) -> None:   # 辅助函数：清某用户的答案缓存
    """用 SCAN 逐批清掉某用户的答案缓存键，保证用例里的第一轮一定未命中。"""
    for key in redis_client.scan_iter(match=f"cache:rag:answer:{user_id}:*", count=200):
        redis_client.delete(key)           # 逐个删除


def _clear_long_memory(user_id: int) -> None:   # 辅助函数：清某用户的长期记忆
    """按 user_id 过滤删除 Milvus 里的长期记忆，并等待删除生效。"""
    import time                            # 导入 time，用于等待 Milvus 落实删除
    import config                          # 导入配置模块，取集合名
    import memory                          # 导入记忆模块，取 Milvus 客户端
    client = memory.get_memory_client()    # 取长期记忆客户端
    name = config.MEMORY_COLLECTION        # 长期记忆集合名
    client.delete(collection_name=name, filter=f"user_id == {int(user_id)}")   # 按用户删
    time.sleep(1)                          # 等 Milvus 落实删除
    client.refresh_load(name)              # 刷新已加载快照，否则删除看不到


def test_cache_hit_still_writes() -> None:
    """测试缓存命中仍写回：MySQL 与 Redis 都要多出这一轮的 user + assistant。

    第 9.7 步的 bug 是命中缓存时直接 return，save_turn 不执行，
    导致该轮消息不落库、不进短期记忆，下一轮 history 缺一段、多轮对话断片。
    """
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    if not _mysql_ready():            # MySQL 不可用
        pytest.skip("MySQL 不可用，跳过")   # 跳过本用例
    if not _redis_ready():            # Redis 不可用
        pytest.skip("Redis 不可用，跳过")   # 跳过本用例
    import json                       # 导入 json，解析短期记忆
    import db                         # 导入数据库模块
    import db_user                    # 导入用户数据模块
    import rag                        # 导入问答模块，直接调 ask 才能看到 cache_hit
    user_id = 8798                    # 专用测试用户编号，避开真实数据
    query = "单元测试用的缓存命中问题"   # 固定问题，方便第二次命中缓存
    redis_client = db.get_redis_client()                    # 取 Redis 客户端
    redis_client.delete(f"mem:short:{user_id}")             # 清短期记忆，保证条数确定
    _clear_user_cache(redis_client, user_id)                # 清缓存，保证第一轮未命中
    conv_id = db_user.create_conversation(user_id, 1, "单元测试 缓存命中写回")   # 建测试会话
    first = rag.ask(query, user_id, 1, conv_id, stream=False)     # 第一轮：未命中，真跑链路
    assert first["cache_hit"] is False, "第一轮不该命中缓存，请检查是否清干净了缓存"
    before = len(_conv_messages(conv_id))                   # 第一轮写完的消息数（应为 2）
    second = rag.ask(query, user_id, 1, conv_id, stream=False)    # 第二轮：同一问题，命中缓存
    assert second["cache_hit"] is True, "第二轮应命中缓存"
    after = len(_conv_messages(conv_id))                    # 第二轮写完的消息数
    assert after - before == 2, f"缓存命中应再写 2 条消息，实际只多了 {after - before} 条"
    roles = [r["role"] for r in _conv_messages(conv_id)][-2:]   # 最后两条的角色
    assert roles == ["user", "assistant"], f"最后两条应为 user + assistant，实际 {roles}"
    items = redis_client.lrange(f"mem:short:{user_id}", 0, -1)  # 短期记忆（最新在最前）
    recent = [json.loads(i)["role"] for i in items[:2]]         # 最近两条的角色
    assert recent == ["assistant", "user"], f"短期记忆最近两条应为 assistant + user，实际 {recent}"
    _drop_conversations([conv_id])                              # 清理会话与消息
    redis_client.delete(f"mem:short:{user_id}")                 # 清理短期记忆
    _clear_user_cache(redis_client, user_id)                    # 清理本轮写的缓存
    if _milvus_ready():                                         # Milvus 可用时顺手清长期记忆
        _clear_long_memory(user_id)                             # 避免测试数据堆积在向量库


def test_cache_hit_writes_long_memory() -> None:
    """测试缓存命中仍写长期记忆：清空后重放同一问题，Milvus 里应能搜到这一轮问答。"""
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    if not _mysql_ready():            # 需要建会话
        pytest.skip("MySQL 不可用，跳过")   # 跳过本用例
    if not _redis_ready():            # 需要清缓存保证第一次未命中
        pytest.skip("Redis 不可用，跳过")   # 跳过本用例
    if not _milvus_ready():           # 需要长期记忆集合
        pytest.skip("Milvus 不可用，跳过")   # 跳过本用例
    import db                         # 导入数据库模块
    import db_user                    # 导入用户数据模块
    import memory                     # 导入记忆模块，用于检索长期记忆
    import rag                        # 导入问答模块
    user_id = 8799                    # 专用测试用户编号
    query = "单元测试用的长期记忆缓存命中问题"   # 固定问题
    redis_client = db.get_redis_client()                    # 取 Redis 客户端
    redis_client.delete(f"mem:short:{user_id}")             # 清短期记忆
    _clear_user_cache(redis_client, user_id)                # 清缓存，保证第一轮未命中
    _clear_long_memory(user_id)                             # 清长期记忆，保证起点为空
    conv_id = db_user.create_conversation(user_id, 1, "单元测试 缓存命中写长期记忆")   # 建会话
    first = rag.ask(query, user_id, 1, conv_id, stream=False)     # 第一轮：未命中
    assert first["cache_hit"] is False, "第一轮不该命中缓存"
    _clear_long_memory(user_id)                             # 再清一次：后面搜到的一定来自第二轮
    second = rag.ask(query, user_id, 1, conv_id, stream=False)    # 第二轮：命中缓存
    assert second["cache_hit"] is True, "第二轮应命中缓存"
    hits = memory.search_long_memory(user_id, query, top_k=3)     # 用问题检索长期记忆
    assert hits, "缓存命中后长期记忆里搜不到内容，说明命中路径没写 Milvus"
    assert any(query in h.get("question", "") for h in hits), f"搜到的记忆不是这一轮：{hits}"
    _drop_conversations([conv_id])                           # 清理会话与消息
    redis_client.delete(f"mem:short:{user_id}")              # 清理短期记忆
    _clear_user_cache(redis_client, user_id)                 # 清理缓存
    _clear_long_memory(user_id)                              # 清理长期记忆


def test_eval_dataset_valid() -> None:
    """测试评测集文件：至少 15 条，每条 question 与 ground_truth 都存在且非空。"""
    import json                       # 导入 json，解析评测集
    import os                         # 导入 os，拼路径
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "eval_dataset.json")   # 评测集路径（项目根/data 下）
    assert os.path.exists(path), f"评测集文件不存在：{path}"     # 断言文件存在
    rows = json.load(open(path, encoding="utf-8"))            # 读并解析
    assert isinstance(rows, list), "评测集应为列表"             # 断言是列表
    assert len(rows) >= 15, f"评测集应至少 15 条，实际 {len(rows)}"   # 断言条数
    for i, row in enumerate(rows, 1):                        # 逐条校验
        assert set(row) >= {"question", "ground_truth"}, f"第 {i} 条字段不全：{list(row)}"   # 字段必须齐
        assert str(row["question"]).strip(), f"第 {i} 条 question 为空"          # 问题非空
        assert str(row["ground_truth"]).strip(), f"第 {i} 条 ground_truth 为空"   # 参考答案非空


def test_eval_result_api() -> None:
    """测试 /eval/result：造一份临时结果文件，断言返回结构与 EvalResp 一致。

    注意：会把 data/ragas_result.json 原文件先备份、测完还原，
    避免把真实评测结果覆盖掉。
    """
    pytest.importorskip("dotenv")     # 缺少 dotenv 依赖时跳过本用例
    import json                       # 导入 json，写临时结果
    import os                         # 导入 os，拼路径
    client = _api_client()            # 创建测试客户端
    if client is None:                # 客户端创建失败
        pytest.skip("FastAPI TestClient 不可用，跳过")   # 跳过本用例
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "ragas_result.json")   # 真实结果文件路径
    backup = open(path, encoding="utf-8").read() if os.path.exists(path) else None   # 先备份原文件
    fake = {                          # 伪造一份结果
        "timestamp": 1700000000,      # 固定时间戳，方便断言
        "model": "test-model",        # 评估模型名
        "embedding": "test-embedding",   # 评估向量模型名
        "summary": {"faithfulness": 0.5, "answer_relevancy": 0.6,      # 四个指标均分
                    "context_precision": 0.7, "context_recall": 0.8},  # 后两个指标
        "details": [{"question": "测试问题", "answer": "测试回答",       # 单条明细
                     "ground_truth": "测试参考答案", "faithfulness": 0.5,     # 前两个分数
                     "answer_relevancy": 0.6, "context_precision": 0.7,       # 中间两个分数
                     "context_recall": 0.8}]}                                 # 最后一个分数
    try:                              # 无论断言是否失败都要还原现场
        open(path, "w", encoding="utf-8").write(json.dumps(fake, ensure_ascii=False))   # 写临时文件
        resp = client.get("/eval/result")     # 请求评测结果
        assert resp.status_code == 200, f"/eval/result 状态码异常：{resp.status_code}"   # 断言状态码
        body = resp.json()            # 解析响应
        for key in ("timestamp", "model", "embedding", "summary", "details"):   # 断言顶层字段齐全
            assert key in body, f"响应缺少字段：{key}"                # 缺字段即失败
        assert body["model"] == "test-model", f"model 未透传：{body['model']}"        # 断言模型名
        assert body["timestamp"] == 1700000000, f"timestamp 未透传：{body['timestamp']}"   # 断言时间戳
        for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            assert name in body["summary"], f"summary 缺少 {name}"    # 断言四项均分齐全
        assert body["summary"]["faithfulness"] == 0.5, f"均分不对：{body['summary']}"   # 断言均分数值
        assert len(body["details"]) == 1, f"details 条数不对：{len(body['details'])}"   # 断言明细条数
        assert body["details"][0]["question"] == "测试问题", "明细 question 未透传"     # 断言明细内容
        assert body["details"][0]["context_recall"] == 0.8, "明细 context_recall 未透传"   # 断言明细分数
    finally:                          # 收尾：还原或删除
        if backup is None:            # 原来没有文件
            if os.path.exists(path):  # 临时文件还在
                os.remove(path)       # 删掉，恢复成"没有文件"的状态
        else:                         # 原来有文件
            open(path, "w", encoding="utf-8").write(backup)   # 把原内容写回去


# ===================== 第 13 步：PDF 列表可配置 + 一键重建 =====================

def test_config_pdf_files_parsed(monkeypatch) -> None:
    """测试 PDF_FILES 解析：逗号分隔、去空白、丢空项。

    说明：config 在 import 时就把环境变量读成了模块常量 `config.PDF_FILES`，
    所以这里直接 monkeypatch 这个常量（它等价于"环境变量设成这个值"），
    再调用 `get_pdf_files()` 验证解析逻辑。
    """
    import config                    # 导入配置模块
    monkeypatch.setattr(config, "PDF_FILES", "a.pdf,b.pdf")          # 两个条目
    assert config.get_pdf_files() == ["a.pdf", "b.pdf"], "普通两个条目解析不对"   # 断言切分结果
    monkeypatch.setattr(config, "PDF_FILES", " a.pdf , ,b.pdf ,")    # 带空白与空项
    assert config.get_pdf_files() == ["a.pdf", "b.pdf"], "空白与空项没被清理"     # 断言清理结果


def test_config_pdf_files_empty(monkeypatch) -> None:
    """测试 PDF_FILES 为空：返回 None，表示"扫 PDF_DIR 全目录"。"""
    import config                    # 导入配置模块
    for raw in ("", "   "):          # 空串与纯空白两种写法
        monkeypatch.setattr(config, "PDF_FILES", raw)                # 逐个设值
        assert config.get_pdf_files() is None, f"空值应返回 None，实际 {config.get_pdf_files()!r}"   # 断言


def test_resolve_pdf_path_relative(monkeypatch) -> None:
    """测试相对条目：只写文件名时按 PDF_DIR 拼接。"""
    import os                        # 导入 os，用于同样的拼接规则比对
    import config                    # 导入配置模块
    monkeypatch.setattr(config, "PDF_DIR", "/data/pdfs")             # 固定一个目录便于断言
    got = config.resolve_pdf_path("a.pdf")                           # 解析相对路径
    assert got == os.path.join("/data/pdfs", "a.pdf"), f"相对路径拼错了：{got}"   # 断言拼接结果


def test_resolve_pdf_path_absolute() -> None:
    """测试绝对路径：带盘符或以斜杠开头的条目原样返回，不再拼 PDF_DIR。"""
    import config                    # 导入配置模块
    for raw in (r"D:\x\a.pdf", "D:/x/a.pdf", "/home/x/a.pdf"):       # 三种绝对路径写法
        assert config.resolve_pdf_path(raw) == raw, f"绝对路径被改写了：{raw}"   # 断言原样返回


def test_parse_all_pdfs_missing_file(monkeypatch) -> None:
    """测试文件不存在时不中断：PDF_FILES 指向不存在的文件，应跳过并返回空列表、不抛异常。"""
    import config                    # 导入配置模块
    import ingest                    # 导入解析模块
    monkeypatch.setattr(config, "PDF_FILES", "这个文件肯定不存在.pdf")   # 指向一个不存在的文件
    results = ingest.parse_all_pdfs()                                    # 不应抛异常
    assert results == [], f"文件不存在时应返回空列表，实际 {results}"       # 断言跳过而非报错


def test_reindex_smoke(monkeypatch) -> None:
    """测试一键重建的编排：三步按顺序被调用；解析出 0 块时中止、不去动 Milvus。"""
    import ingest                    # 导入解析模块（其函数将被替换）
    import enrich                    # 导入增强模块
    import vector_store              # 导入向量库模块
    import reindex                   # 导入一键重建模块
    calls = []                       # 记录调用顺序

    def fake_parse():                # 假的解析：1 个 PDF、3 个块
        calls.append("parse")        # 记一次调用
        return [{"chunks": [1, 2, 3]}]   # 返回带 chunks 的结果
    monkeypatch.setattr(ingest, "parse_all_pdfs", fake_parse)              # 替换解析
    monkeypatch.setattr(ingest, "save_chunks_to_json",
                        lambda r: calls.append("save") or "fake.json")     # 替换落盘
    monkeypatch.setattr(enrich, "enrich_all",
                        lambda: calls.append("enrich") or {"final": 2})    # 替换增强
    monkeypatch.setattr(vector_store, "load_chunks_from_json",
                        lambda: calls.append("load") or [{"a": 1}, {"a": 2}])   # 替换读增强结果
    monkeypatch.setattr(vector_store, "insert_chunks",
                        lambda items, recreate=False: calls.append("insert") or len(items))   # 替换入库

    result = reindex.run_reindex()                                        # 执行编排
    assert calls == ["parse", "save", "enrich", "load", "insert"], f"调用顺序不对：{calls}"   # 断言顺序
    assert result["ok"] is True, f"应成功：{result}"                       # 断言整体成功
    assert result["steps"] == ["ingest", "enrich", "index"], f"步骤名不对：{result['steps']}"   # 断言三步
    assert result["detail"]["ingest"]["out"] == 3, "解析输出块数没统计对"    # 断言解析输出数
    assert result["detail"]["index"]["out"] == 2, "入库条数没统计对"        # 断言入库条数

    # 第二段：解析出 0 块时必须中止，不能拿空数据去重建 Milvus（否则会把索引清空）
    calls.clear()                                                         # 清空调用记录
    monkeypatch.setattr(ingest, "parse_all_pdfs", lambda: calls.append("parse") or [])   # 空结果
    blocked = reindex.run_reindex()                                       # 再跑一次
    assert blocked["ok"] is False, "解析出 0 块时应判为失败"                 # 断言失败
    assert blocked["failed"] == "ingest", f"应停在解析这步：{blocked['failed']}"   # 断言停在哪步
    assert "insert" not in calls, "0 块时不该继续去重建 Milvus"              # 断言没动 Milvus
