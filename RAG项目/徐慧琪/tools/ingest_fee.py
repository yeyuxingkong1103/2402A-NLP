"""收费口径语料入库：解析 → 切分 → 编码 → 写入 fee_corpus。

存在的理由：费用区间必须能指回片段（技术方案 6.4 硬规矩），所以语料要
按「可被引用」的粒度切 —— 与法条同为条款级，而不是整篇一个块。

与法条入库的两处不同（都是**有意**的）：
  ①schema 是本文件自己的（law_chunks 的 FIELDS 把 law_id / article_no /
    chunk_type 写死在 app/db/milvus.py 里，那套只服务法条；生产集合有
    3388 条真实数据，不为兼容第二者去动它）；
  ②切分只按空行。语料（4 份律所收费标准 + 司发通〔2021〕87 号）已按
    「一个语义单元一段、段间空行」排版，brief 预备的「第X条」正则分支
    不适用：律所标准没有条款号结构，硬套会把「一、计时收费」判成无编号。
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import sys

# 入库脚本在 tools/ 下而 app 包在 backend/ 下，与 tools/smoke_retrieval.py 同写法
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from pymilvus import DataType  # noqa: E402

# 编码与向量索引参数、upsert 与计数**全部**复用既有实现，不另写一套：
# 另写会让「索引参数 / 重试口径 / 计数口径」在两处各存一份，改一处漏一处
from app.db.milvus import (  # noqa: E402
    DENSE_DIM, DENSE_INDEX, SPARSE_INDEX, entity_count, get_client, insert_chunks,
)
from app.ingest.embed import encode_texts, load_model  # noqa: E402
# 检索面（search）与集合身份住在生产包里，服务进程与 CLI 用的是同一份实现 ——
# 搬家的理由见那个模块的 docstring（tools/ 不在服务进程的 sys.path 上）。
# 下面**再导出**它们不是留第二份实现，而是本模块的公开面早已被 tools/tests 冻结
# （FEE_COLLECTION / FEE_FILTER_EXPR / search …），换掉导入路径就断了那批用例 ——
# 移动实现不该顺手改契约
from app.recommend.fee_search import (  # noqa: E402
    FEE_COLLECTION, FEE_FILTER_EXPR, FEE_OUTPUT_FIELDS, FEE_TOP_K, search,
)

FEE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "fee"

# 收费口径语料都是现行文件（司发通〔2021〕87 号未被废止，三份律所标准按
# 年度备案公示）。将来若收进失效文件，这列的取值要跟着扩
STATUS_VALID = "现行有效"

# 文件名（不含扩展名）→ 短标识 / 地区 / 生效日。
#
# 短标识进主键，是因为 Milvus 的 VARCHAR 长度按**字节**算：最长的 stem 已 90 字节，
# 中文文件名直接进主键会超长；短标识还让人一眼看出某行来自哪份文件。
# region 与 effective_date 的取法：三份律所公示标准都在北京（zjfw.beijing.gov.cn
# 公示），87 号是国家部门文件（全国）。昌民那份**文件自身未标日期**，取其在该平台
# 的公示日（下载 URL 的日期段 2025-10-31）—— 该字段供将来的时效过滤用，取公示日
# 是可接受的近似，但不能当成「文件写明的生效日」。来源与例外详见
# data/raw/fee/_sources/README.md
SOURCES: dict[str, dict[str, str]] = {
    "司法部等三部门关于规范律师服务收费的意见（司发通〔2021〕87号）":
        {"code": "sf87", "region": "全国", "effective_date": "2021-12-26"},
    "北京智深律师事务所收费标准":
        {"code": "bjzs", "region": "北京", "effective_date": "2025-01-01"},
    "北京昌民律师事务所收费标准":
        {"code": "bjcm", "region": "北京", "effective_date": "2025-10-31"},
    "北京融理律师事务所收费标准":
        {"code": "bjrl", "region": "北京", "effective_date": "2023-01-01"},
}

# fee_corpus 自己的 schema：设计文档 §五 定的六个业务字段 + 主键 + 双向量。
# 主键名沿用 chunk_id 是**有意**的耦合：app/db/milvus.entity_count 把
# `chunk_id != ''` 写死成 count(*) 的过滤条件，改名就得去动那个模块（连带
# 碰 law_chunks 的既有行为与测试），而换个名字换不来任何东西，故从众。
FEE_FIELDS: list[tuple[str, DataType, dict]] = [
    ("chunk_id", DataType.VARCHAR, {"is_primary": True, "max_length": 32}),
    ("dense", DataType.FLOAT_VECTOR, {"dim": DENSE_DIM}),
    ("sparse", DataType.SPARSE_FLOAT_VECTOR, {}),
    ("source_doc", DataType.VARCHAR, {"max_length": 255}),
    ("source_no", DataType.VARCHAR, {"max_length": 32}),
    ("region", DataType.VARCHAR, {"max_length": 16}),
    ("effective_date", DataType.VARCHAR, {"max_length": 10}),
    ("status", DataType.VARCHAR, {"max_length": 16}),
    # 实测最长 2075 字节（智深含价目表的民事段，2026-09-29 语料重建后重测；
    # 重建前最长 1830），留近 2 倍余量。
    # Milvus 定短了是**静默截断**（law_chunks 的 text 也是这么踩过的），
    # 而截掉的可能正是费用数字 —— 留足余量比省几个字节重要
    ("text", DataType.VARCHAR, {"max_length": 4000}),
]

# 标量索引：status 与法条同口径（检索一律带效力过滤是既定习惯，见 FEE_FILTER_EXPR），
# region 是本语料唯一的地区轴（三份北京 + 一份全国）。region 本期检索不滤它
# （设计 §五：用户问句通常不含地区），但索引成本随数据量线性，语料再扩十倍也只有
# 几百行，先建上免得将来漏建
FEE_SCALAR_INDEX_FIELDS = ["status", "region"]

# 小标题编号：条号 / 汉字序号 / 括号汉字序号 / 阿拉伯序号 / 小数点式序号。
# 末两组带 (?!\d) 是为了不把「7.4」咬成「7.」——小数点的第二个点也是分隔符类字符
_NUMBER_MARKER = (r"第[一二三四五六七八九十百零]+条"
                  r"|[一二三四五六七八九十]+、"
                  r"|（[一二三四五六七八九十]+）"
                  r"|\d+[、.．](?!\d)"
                  r"|\d+[.．]\d+")

# 编号后紧跟单位词 = 那是金额/费率而不是序号。语料原文「7.4 万元+标的额 100 万元
# 以上部分的4%」是价目表的半截（PDF 文本层把表切断了），7.4 是费率：
# 当成序号会让「依据：北京智深… 7.4」指向一个不存在的编号
_NOT_MONEY = r"(?!\s*(?:万元|元|%|％))"

_HEADING = re.compile(f"^(?:{_NUMBER_MARKER}){_NOT_MONEY}")


def split_paragraphs(text: str) -> list[str]:
    """按空行切段并去掉空白块。切分口径就是入库粒度，故只此一招。

    为什么不按内容二次切：语料已按「一个语义单元一段」排版，段内是完整的
    价目或标准；再按「第X条」切会把「1． 10 万元以下…2． 10 万元以上…」
    这类同一张价目表拆碎，检索命中后指回的片段就缺了上下文，而费用回查
    恰恰要靠片段里的数字（技术方案 6.4 硬规矩①）。
    """
    return [chunk.strip() for chunk in re.split(r"\n\s*\n", text or "")
            if chunk.strip()]


def extract_source_no(unit: str) -> str:
    """取片段开头的小标题编号（无编号返回空串）。始终是 str。

    只取编号、不取标题正文：真实语料里编号后面常直接是一整句话
    （「（一）提升律师服务收费合理化水平。律师服务收费项目…」），
    取正文会让 source_no 变成一句长文，而渲染方是拿它拼「依据：文件 编号」的。
    空串是合法的退化值：渲染方 `if p` 会跳过它，不印出半截「依据」。

    str 而不是 int（法条的 article_no 是 int）是**硬契约**：tools/ask.py
    对它做 " ".join(...)，非 str 会在渲染期抛 TypeError，而那时答案已算好。
    """
    matched = _HEADING.match(unit)
    return matched.group(0) if matched else ""


def source_code(stem: str) -> str:
    """文件的短标识：SOURCES 里有就用登记的，没有则退化为 stem 的哈希前缀。

    退化分支让 build_rows 能处理任意文件（测试里的临时文件、将来新增的语料），
    且主键长度恒定、不含中文 —— 不依赖调用方先登记。
    """
    if stem in SOURCES:
        return SOURCES[stem]["code"]
    return hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]


def build_rows(doc_path: pathlib.Path, region: str, effective_date: str) -> list[dict]:
    """把一份文档切成待入库的行（不含向量，向量由 encode_rows 统一补）。

    主键显式给且短（`代号-序号`），不用 Milvus 的 auto_id：auto_id 会让重跑
    生成一份新 id、旧数据原样保留（集合从 26 变 52 且不报错），而本设计要的
    正是「重跑即覆盖」的幂等。序号在文件内单调，重跑同一份语料得到同一批主键。
    """
    text = doc_path.read_text(encoding="utf-8")
    code = source_code(doc_path.stem)
    rows: list[dict] = []
    for index, unit in enumerate(split_paragraphs(text)):
        # source_doc 取亲本名主干：它是渲染方拼「依据」的**唯一**溯源字符串
        # （fee_generation_log 的列里没有它），必须人人可查
        rows.append({
            "chunk_id": f"{code}-{index:03d}",
            "text": unit,
            "source_doc": doc_path.stem,
            "source_no": extract_source_no(unit),
            "region": region,
            "effective_date": effective_date,
            "status": STATUS_VALID,
        })
    return rows


def ensure_collection(client) -> None:
    """集合不存在才建。重跑入库走这个入口，不会误删已有数据。

    这里是 app/db/milvus.create_collection 的 fee 版：那边 schema 与索引都
    按 law_chunks 的 FIELDS 拼死了，传 name 只换集合名、换不掉字段。所以
    本文件自带一份 schema，而**索引参数与向量维度仍从那边取**（DENSE_INDEX /
    SPARSE_INDEX / DENSE_DIM），只此一处不同源会被这里改漏。
    auto_id 与 enable_dynamic_field 都显式关掉，理由同那边：动态字段会让
    拼错的字段名静默写进库而不报错，属于「看起来成功」的失败。
    """
    if client.has_collection(FEE_COLLECTION):
        return
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    for name, dtype, params in FEE_FIELDS:
        schema.add_field(name, dtype, **params)
    index = client.prepare_index_params()
    index.add_index("dense", **DENSE_INDEX)
    index.add_index("sparse", **SPARSE_INDEX)
    for name in FEE_SCALAR_INDEX_FIELDS:
        index.add_index(name, index_type="INVERTED")
    client.create_collection(FEE_COLLECTION, schema=schema, index_params=index)


def iter_docs(doc_dir: pathlib.Path = FEE_DIR) -> list[pathlib.Path]:
    """列出语料文件：排除 `_sources/` 与任何下划线开头的条目。

    下划线是本目录的约定（`_sources/` 存原始下载件与来源说明、`_probe_*` 是
    未入选的调研件）。不排就会把「来源说明」当成收费口径灌进库 —— 检索照样
    命中，只是指回的是一段说明文字，还不报错。
    """
    return sorted(p for p in doc_dir.glob("*.txt")
                  if p.is_file() and not p.name.startswith("_"))


def encode_rows(rows: list[dict], model) -> list[dict]:
    """给每行补上 dense / sparse，返回新行列表（不改原行）。

    编码走 app/ingest/embed.encode_texts，不另写：它保证「条数与顺序与输入
    严格对齐」并在不符时抛错 —— 错位是静默的，会把 A 段的向量写到 B 段上。
    """
    encoded = encode_texts(model, [row["text"] for row in rows])
    return [{**row, "dense": dense, "sparse": sparse}
            for row, (dense, sparse) in zip(rows, encoded)]


def main(client=None, model=None, doc_dir: pathlib.Path = FEE_DIR) -> dict[str, int]:
    """端到端：读目录 → 切分 → 编码 → 写 fee_corpus，返回统计。

    client 与 model 可注入：真跑时省参（自建连接、加载 bge-m3），离线测试
    塞替身 —— 本函数测的是编排，不是 Milvus 或模型的实现。
    """
    own_client = client is None
    if own_client:
        client = get_client()
    try:
        docs = iter_docs(doc_dir)
        if not docs:
            # 空目录会产出 0 行，而「实体数 0 == 行数 0」能让下面的校验通过：
            # 不拦就是「成功地把库灌成空」，且全链路不报任何错
            raise RuntimeError(f"{doc_dir} 下没有语料文件，确认语料是否放对位置")
        ensure_collection(client)
        rows: list[dict] = []
        for path in docs:
            # KeyError 是刻意的（与 app/ingest/embed 的 meta 查表同口径）：
            # region / effective_date 是 build_rows 的入参，新文件必须先在
            # SOURCES 里登记。静默给空串会让这层元数据整个失去意义
            meta = SOURCES[path.stem]
            rows.extend(build_rows(path, meta["region"], meta["effective_date"]))
        if model is None:
            model = load_model()
        written = insert_chunks(client, encode_rows(rows, model), FEE_COLLECTION)
        entities = entity_count(client, FEE_COLLECTION)
        # 实体数 == 本次行数 才是幂等的证明：主键若被换成 auto_id，重跑会翻倍；
        # 语料若删过文件，旧行会留下而实体数偏大。两种都当场中断 —— 静默继续
        # 等于把过期口径留到检索阶段，而检索不会报错，只会返回一段过期价位
        if entities != len(rows):
            raise RuntimeError(
                f"集合 {FEE_COLLECTION} 实体数 {entities} ≠ 本次入库 {len(rows)} 条："
                "语料与集合内容已不一致（删过文件或换过主键口径），请核对后重灌")
        return {"docs": len(docs), "rows": len(rows),
                "written": written, "entities": entities}
    finally:
        if own_client:
            client.close()


if __name__ == "__main__":
    stats = main()
    print(f"入库完成：{stats}")
    print(f"集合 {FEE_COLLECTION} 实体数 {stats['entities']}，"
          f"build_rows 合计 {stats['rows']} 条")
