"""tests/unit/test_new_architecture.py —— src/ 新架构的纯函数单元测试。

在链路中的位置：
    单元测试层（tests/unit/），对应 tests/integration/ 那层。
    **不依赖任何外部服务**：不连 Milvus、不调 Ollama、不起 FastAPI。
    因此这组测试是"改了逻辑立刻能跑一遍"的最快反馈回路。

覆盖 src/ 里三个最适合单测的纯函数：
    chunkers.build_chunks          分块（结构与元数据）
    query_transform                改写与指代消解
    postprocess                    答案清洗与引用生成

用 pytest 风格（裸 assert）而不是 unittest 风格：
    新架构这边的测试统一用 pytest 写，断言更简洁，
    失败时的输出（pytest 会展示变量实际值）也更直观。
"""
from src.offline.chunkers import build_chunks
from src.online.query_transform import expand_query, resolve_coreference
from src.online.postprocess import clean_answer, references


def test_chunkers_build_semantic_chunks():
    """验证 semantic 策略能切出块，且块上带着页码和摘要。

    输入用 Markdown 标题 + 两个段落，走 semantic 策略
    （它内部先用段落分块，再给每块粘上上一段的尾部）。

    三处断言各自的用意：
        assert chunks                     切出了内容（不是空列表）
        chunks[0].page == 1               页码被正确传递下来 ——
                                          这是答案引用"第 N 页"的来源，丢不得
        chunks[0].summary 非空            摘要被生成 ——
                                          知识库列表页靠它展示，为空会导致界面空白

    metadata 传 {} 而不是省略：
        build_chunks 会把 page 的 metadata 合并进 chunk，
        传空字典模拟"解析器没提供额外元数据"的正常情形。
    """
    chunks = build_chunks([{"page": 1, "text": "# 标题\n第一段。\n\n第二段。", "metadata": {}}], strategy="semantic")
    assert chunks
    assert chunks[0].page == 1
    assert chunks[0].summary


def test_query_transform_expands_law_terms():
    """验证法律类术语扩展与指代消解。

    第一条断言（术语扩展）：
        问"合同赔偿怎么处理"，期望改写结果里出现"证据"。
        规则表把"合同"映射到 [违约, 条款, 履行, 解除]、把"赔偿"映射到
        [责任, 损失, 证据, 法律依据] —— "证据"正是从这里补进来的。
        它存在的意义是让字面匹配路也能命中文档里的原词。

    第二条断言（指代消解）：
        历史里有"合同违约"，本轮问"这个怎么办"（以代词开头），
        期望消解结果以"合同违约"开头 —— 即上一句被拼到了前面。
        验证的是"追问能被正确还原成完整问题"，
        否则"这个怎么办"这种查询在检索时毫无意义。

    用 startswith 而不是相等：
        消解结果是 "合同违约；追问：这个怎么办" 这种拼接形式，
        断言只关心"上一句被放到了开头"这个关键性质，
        不锁死冒号、分号这类格式细节，让格式微调不会误报失败。
    """
    bundle = expand_query("合同赔偿怎么处理")
    assert "证据" in bundle.rewritten
    assert resolve_coreference("这个怎么办", [{"speaker": "user", "content": "合同违约"}]).startswith("合同违约")


def test_postprocess_references():
    """验证答案清洗与引用生成。

    第一条断言（清洗）：
        输入 "<think>x</think>回答"（模拟推理型模型的输出格式），
        期望结果恰好等于 "回答" —— 思考过程被整段剥掉了。
        这里用 == 全等断言（而不是 assertIn），是有意的：
        要验证的正是"除了答案什么都不剩"，用包含判断会漏掉残留的噪声。

    第二条断言（引用生成）：
        输入一条 Milvus 风格的嵌套记录 {"entity": {...}}，
        期望引用列表的第一项 source == "a.txt"。
        这验证了 references 的字段兜底逻辑确实能处理 entity 嵌套结构 ——
        取错字段的话引用会是空的或 "unknown"，
        用户就无法核对答案依据了。
    """
    answer = clean_answer("<think>x</think>回答")
    assert answer == "回答"
    refs = references([{"entity": {"doc_source": "a.txt", "page": 1, "content": "abc"}}])
    assert refs[0]["source"] == "a.txt"
