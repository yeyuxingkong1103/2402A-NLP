# 四关校验是本系统的安全阀，它的测试必须是"能失败的测试"：
# 每关都配一条应当被拒绝的用例。假语料只用来喂查询器，
# 被校验的答案形态（条号写法、quote 片段）全部取自真实法条。
from app.generation.cite_verify import (
    MIN_QUOTE_LEN, normalize_quote, verify_answer,
)
from app.generation.schema import Answer

# 第584条原文（节选，取自 data/parsed/law_articles.jsonl 的实读值）
ART_584 = ("第五百八十四条 当事人一方不履行合同义务或者履行合同义务不符合约定，"
           "造成对方损失的，损失赔偿额应当相当于因违约所造成的损失，"
           "包括合同履行后可以获得的利益。")
# 夹具把该条设为两款：款号 1、2 判存在、3 判不存在。实读值见 584 行，
# 但真库（Milvus）里 584 只有 paragraph_no=1 一个款级子块——夹具是假语料，
# 只喂查询器用，款号真假的判定逻辑与真实条有几款无关
PARA_584 = {"paragraphs": {1, 2}, "items": set()}
STATUS_OK = "现行有效"


def _fake_article_of(mapping):
    return mapping.get


def _fake_para_of(mapping):
    return mapping.get


def _corpus():
    return {
        "article_of": _fake_article_of({
            584: {"text": ART_584, "status": STATUS_OK, "law_id": "minfadian"},
            1042: {"text": "第一千零四十二条 禁止包办、买卖婚姻……", "status": "已废止",
                   "law_id": "minfadian"},
        }),
        "para_index_of": _fake_para_of({584: PARA_584}),
    }


def _answer(article="第五百八十四条", quote=ART_584[7:30], **kwargs):
    """构造一条形如模型输出的答案。quote 默认取真实原文片段。"""
    return Answer.model_validate({
        "status": "ok", "answer": "应当赔偿。",
        "citations": [dict({"law": "中华人民共和国民法典", "article": article,
                            "paragraph": None, "item": None, "quote": quote}, **kwargs)],
    })


def test_valid_citation_passes_all_gates():
    assert verify_answer(_answer(), **_corpus()) == []


def test_rejects_answer_without_citations():
    # AC-6：没有引用的结论一律不可信——不可信就当幻觉处理
    empty = Answer.model_validate({"status": "ok", "answer": "就这样。", "citations": []})
    failures = verify_answer(empty, **_corpus())
    assert failures and "引用" in failures[0]


def test_gate1_rejects_nonexistent_article():
    """第一关：条号在库里不存在。这是最典型的幻觉。"""
    failures = verify_answer(_answer(article="第九千九百九十九条"), **_corpus())
    assert any("不存在" in f for f in failures)


def test_gate2_rejects_repealed_article():
    """第二关：条存在但已废止。FR-3.3 要求废止条文绝不能被引用。"""
    failures = verify_answer(_answer(article="第一千零四十二条"), **_corpus())
    assert any("效力" in f for f in failures)


def test_gate3_rejects_nonexistent_paragraph():
    """第三关：款号不存在。纯子串匹配抓不住这个。"""
    failures = verify_answer(_answer(paragraph=3), **_corpus())
    assert any("款" in f for f in failures)


def test_gate3_accepts_existing_paragraph():
    assert verify_answer(_answer(paragraph=2), **_corpus()) == []


def test_gate3_rejects_nonexistent_item():
    failures = verify_answer(_answer(item="三"), **_corpus())
    assert any("项" in f for f in failures)


def test_gate3_accepts_parenthesized_item():
    # 设计文档 4.3：Milvus 的 item_no 存光杆中文数字（存「一」不存「（一）」），
    # 模型按文书习惯写「（一）」时必须剥括号后比对，否则正确引用被误杀
    corpus = _corpus()
    corpus["para_index_of"] = _fake_para_of({584: {"paragraphs": {1, 2}, "items": {"一"}}})
    assert verify_answer(_answer(item="（一）"), **corpus) == []


def test_gate3_single_paragraph_article_treats_first_paragraph_as_legal():
    # 库里该条查不到款级子块时（子块缺失），实务上整条即一款：
    # 「第一款」合法，但「第二款」及以后仍要拒——这是第③关的边界
    corpus = _corpus()
    corpus["para_index_of"] = _fake_para_of({584: {"paragraphs": set(), "items": set()}})
    assert verify_answer(_answer(paragraph=1), **corpus) == []
    assert any("款" in f for f in verify_answer(_answer(paragraph=2), **corpus))


def test_gate4_rejects_rewritten_quote():
    """第四关：模型改写了原文（哪怕意思一样）。"""
    failures = verify_answer(_answer(quote="不履行合同义务造成损失的应当赔偿全部损失"), **_corpus())
    assert any("原文" in f for f in failures)


def test_gate4_rejects_too_short_quote():
    # 短片段（如"当事人"）能骗过子串匹配，设长度下限正是为了堵这个
    failures = verify_answer(_answer(quote="当事人"), **_corpus())
    assert any("原文" in f or "短" in f for f in failures)


def test_gate4_tolerates_whitespace_differences():
    # 模型把原文里的换行/空格改动是常见且无害的，不该判失败。
    # 片段取到「，」为止（[7:34]）：不含标点的片段会让 replace 空转、
    # 这条用例恒真，所以下面的前提断言必须留着
    quote_with_spaces = ART_584[7:34].replace("，", "，\n  ")
    assert quote_with_spaces != ART_584[7:34], "构造不出带空白的引文，用例前提不成立"
    assert verify_answer(_answer(quote=quote_with_spaces), **_corpus()) == []


def test_article_number_written_in_arabic_is_accepted():
    # 引用写法不统一是常态，归一化后必须等价
    assert verify_answer(_answer(article="584"), **_corpus()) == []


def test_unparseable_article_raises_failure():
    failures = verify_answer(_answer(article="公司法相关条款"), **_corpus())
    assert any("无法解析" in f for f in failures)


def test_normalize_quote_strips_all_whitespace():
    assert normalize_quote(" 当事 人\n一方　未支付 ") == "当事人一方未支付"


def test_min_quote_len_is_ten():
    assert MIN_QUOTE_LEN == 10


def test_all_failures_are_collected_not_just_the_first():
    # 一次列出全部问题，重生成时能把所有原因一并反馈给模型，
    # 只报第一条会让重生成反复撞同一堵墙
    bad = _answer(article="第九千九百九十九条", quote="短")
    assert len(verify_answer(bad, **_corpus())) >= 2


def test_corpus_article_delegates_to_fetch_articles(monkeypatch):
    # 条原文与效力的取值只许有一条来源：Task 2 的 fetch_articles（它的列绑定
    # 由 test_article_lookup 的 SQL 文本断言守门）。corpus 里另写一份 SQL 就是
    # 绕开那道断言，所以这里钉死「article() 走 fetch_articles 且按需投影」。
    # 局部导入：cite_verify 的纯函数用例不该因为本模块而拖上数据库依赖
    from app.generation import corpus as corpus_mod

    seen = []

    def fake_fetch(conn, nos, law_id):
        seen.append((conn, list(nos), law_id))
        if 9999 in nos:
            return []
        return [{"text": "条原文", "status": STATUS_OK, "law_id": law_id, "chunk_id": None}]

    monkeypatch.setattr(corpus_mod, "fetch_articles", fake_fetch)
    law = corpus_mod.LawCorpus("conn-sentinel", None, law_id="minfadian")
    assert law.article(584) == {"text": "条原文", "status": STATUS_OK, "law_id": "minfadian"}
    assert law.article(9999) is None
    # 连接、条号、law_id 都必须原样透传给 fetch_articles，不许多查也不许改写
    assert seen == [("conn-sentinel", [584], "minfadian"),
                    ("conn-sentinel", [9999], "minfadian")]


def test_corpus_para_index_counts_only_paragraph_rows_not_parse_lines():
    # 真库实测（第395条）：2 款 + 7 项，款行落在解析行号 1 与 9、项行占 2..8。
    # 照行的 paragraph_no 建 paragraphs 集合会让第 3~9 款全判「存在」，第③关
    # 在这 91 条含项法条上退化（旧夹具喂的是语义款号，测不出这个真形态）。
    # 款号的语义是「第几款」，只能数 chunk_type=paragraph 的行、按 1..N 归一
    class FakeMilvus:
        def query(self, name, filter, output_fields):
            return [{"chunk_type": "paragraph", "paragraph_no": 1, "item_no": None},
                    {"chunk_type": "paragraph", "paragraph_no": 9, "item_no": None},
                    {"chunk_type": "item", "paragraph_no": 2, "item_no": "一"},
                    {"chunk_type": "item", "paragraph_no": 8, "item_no": "七"}]

    from app.generation.corpus import LawCorpus
    law = LawCorpus("conn-sentinel", FakeMilvus())
    assert law.para_index(395) == {"paragraphs": {1, 2}, "items": {"一", "七"}}
    # 判定侧同链复核：第 2 款合法、第 3 款必须被拒（改前这里会放行）
    corpus = _corpus()
    corpus["para_index_of"] = law.para_index
    assert verify_answer(_answer(paragraph=2), **corpus) == []
    assert any("款" in f for f in verify_answer(_answer(paragraph=3), **corpus))
