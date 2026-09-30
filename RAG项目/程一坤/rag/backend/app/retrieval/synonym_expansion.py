"""法律术语同义扩写（批次 13）。

问题：用户口语表述与法条用语差距太大时，关键词（BM25）路会整条召不到。
实测（批次 11 诊断）：confuse-049「退休以后被公司返聘」对应解释（一）第32条
的「已经依法享受养老保险待遇…按劳务关系处理」；asof-031「约定不缴社保」对应
劳动合同法第46条第三项的「未依法为劳动者缴纳社会保险费」——两路都没进池。

做法（用户裁决的案 1，单向、可回退）：
  · 只在关键词路做「口语 → 法条用语」单向扩展：命中替代表达就把该组的
    法条用语以 OR 方式追加进 BM25 查询，原词保留；
  · 向量路与重排输入一律不动（bge-m3 对口语本就鲁棒，不该被扩展词稀释）；
  · 术语表落盘 data/（带版本号），本模块只加载不硬编码，便于增补。

四道安全阀：
  1. 易混术语对禁止交叉扩展（经济补偿 ↔ 赔偿金、解除 ↔ 终止）；
  2. ASCII 术语整词匹配（"2N" 不触发 N 组）；
  3. 负向规则：非劳动争议词命中即整条 query 不扩展（保护拒答题）；
  4. 配置开关默认关（SYNONYM_EXPANSION_ENABLED）。

为什么"单向"：反向（法条用语 → 口语）会让 BM25 查询掺入大量口语词，
把精确的条文用词命中率稀释掉；而"口语召不到"才是真问题，所以只修单向。
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import PROJECT_ROOT_DIRECTORY

# 术语表默认位置：项目根 data/（不进 backend/，便于非开发同学增补）
DEFAULT_TABLE_PATH = PROJECT_ROOT_DIRECTORY / "data" / "legal_synonyms_v1.json"

# 纯 ASCII 术语（N / 2N 之类）走词边界匹配，中文术语走子串匹配
ASCII_TERM_PATTERN = re.compile(r"^[0-9A-Za-z]+$")
WHITESPACE_PATTERN = re.compile(r"[\s\u3000]+")


class SynonymTableError(RuntimeError):
    """术语表缺失或格式不合法。

    单独定义类型：术语表是"人工维护的外部数据"，其错误要能和"检索链路故障"
    区分开 —— 前者是数据问题（应报错给人改表），后者才该重试/降级。
    """


@dataclass(frozen=True)
class SynonymGroup:
    """一组同义术语：法条用语 + 替代表达 + 条文出处。

    字段：id 组号（诊断与日志定位用）；standard 法条用语（扩展时追加进 BM25）；
    alternatives 用户可能说的口语/替代表达（命中判据）；source 条文出处（人工核对用）。

    为什么用 tuple 且 frozen：术语表加载后视为只读快照，避免运行中有人就地改
    某组词导致"同一进程内扩展结果随时间变化"（排查检索差异时极其难查）。
    """

    id: int
    standard: tuple[str, ...]
    alternatives: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class SynonymTable:
    """术语表（含版本号）。

    字段：version 术语表版本；direction 扩展方向（应恒为单向"口语→法条用语"）；
    groups 全部术语组；confusable_pairs 易混术语对（安全阀 1）；
    negative_keywords 负向关键词（安全阀 3）。
    """

    version: str
    direction: str
    groups: tuple[SynonymGroup, ...]
    confusable_pairs: tuple[tuple[str, ...], ...] = ()
    negative_keywords: tuple[str, ...] = ()


@dataclass
class ExpansionResult:
    """一次扩展的结果，供检索与诊断复用。

    字段：phrases 实际追加进 BM25 的法条用语；
    matches 命中明细 (组号, 命中的替代表达, 追加的法条用语)；
    skipped 被安全阀挡下的词条及原因（诊断"为什么没扩展"靠它）。

    为什么这个类**不加 frozen**：它是"本次调用的累积容器"，explain() 一路往里
    append；前两个类是术语表快照，性质不同。
    """

    phrases: list[str] = field(default_factory=list)
    # (group_id, 命中的替代表达, 追加的法条用语)
    matches: list[tuple[int, str, str]] = field(default_factory=list)
    # 被安全阀挡下的词条及原因，用于诊断"为什么没扩展"
    skipped: list[str] = field(default_factory=list)


def load_synonym_table(table_path: str | Path = DEFAULT_TABLE_PATH) -> SynonymTable:
    """加载并校验术语表；结构不对就明确报错，不静默降级。

    参数：table_path —— 术语表 JSON 路径（默认 data/legal_synonyms_v1.json）。
    返回：SynonymTable 快照。
    异常：SynonymTableError —— 文件不存在 / 非法 JSON / 某组缺 standard 或
          alternatives / 整表没有任何组。

    为什么不静默降级成"空表"：静默空表会让"扩写没生效"表现为"召回率莫名下降"，
    归属极难定位；宁可启动即报错，让问题在最早、最显眼的时刻暴露。
    """
    path = Path(table_path)
    if not path.is_file():
        raise SynonymTableError(f"术语表不存在：{path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SynonymTableError(f"术语表不是合法 JSON：{path}（{error.msg}）") from error

    groups: list[SynonymGroup] = []
    for item in payload.get("groups", []):
        standard = tuple(item.get("standard") or ())
        alternatives = tuple(item.get("alternatives") or ())
        # 缺任一侧的组在扩展逻辑里没有意义（既无法命中、也无词可追加），
        # 直接报错指出是哪一组，避免人工改表时漏字段
        if not standard or not alternatives:
            raise SynonymTableError(
                f"术语表第 {item.get('id')} 组缺 standard 或 alternatives"
            )
        groups.append(
            SynonymGroup(
                id=int(item.get("id", 0)),
                standard=standard,
                alternatives=alternatives,
                source=item.get("source", ""),
            )
        )
    if not groups:
        raise SynonymTableError(f"术语表没有任何术语组：{path}")

    # 缺省容忍：负向规则与易混对是"加强项"，未配置时按空处理（功能照常）
    negative = payload.get("negative_rules") or {}
    return SynonymTable(
        version=str(payload.get("version") or ""),
        direction=str(payload.get("direction") or ""),
        groups=tuple(groups),
        confusable_pairs=tuple(
            tuple(pair.get("terms") or ()) for pair in payload.get("confusable_pairs", [])
        ),
        negative_keywords=tuple(negative.get("keywords") or ()),
    )


class SynonymExpander:
    """把口语表述映射成法条用语，供关键词检索扩展使用。"""

    def __init__(self, table_path: str | Path = DEFAULT_TABLE_PATH) -> None:
        """加载术语表并缓存负向关键词。

        参数：table_path —— 术语表路径（默认取 data/ 下的 v1 表）。
        返回：None。

        把 negative_keywords 提到实例属性：explain() 每个 query 都会查它，
        避免每次都穿透 table 属性（也是后续换成预编译正则时的落点）。
        """
        self.table = load_synonym_table(table_path)
        self.negative_keywords = self.table.negative_keywords

    @property
    def version(self) -> str:
        """术语表版本号（写进检索统计便于追溯）。"""
        return self.table.version

    def expand(self, query: str) -> list[str]:
        """返回需要追加进 BM25 查询的法条用语（不含原 query 词条）。

        参数：query —— 用户原始问题。
        返回：法条用语列表；无命中或被安全阀拦下时返回空列表。

        调用方拿到的是纯列表（不关心命中过程）；需要明细请用 explain()。
        """
        return self.explain(query).phrases

    def explain(self, query: str) -> ExpansionResult:
        """扩展并给出命中/跳过明细，供诊断脚本贴证据。

        参数：query —— 用户原始问题。
        返回：ExpansionResult（phrases / matches / skipped）。

        先做"去空白压缩"再匹配：用户输入里的空格、全角空格会让 "2 N" 这类
        写法漏判；压缩后 ASCII 术语仍用词边界判（见 _matches），不会误伤 "2N"。
        """
        result = ExpansionResult()
        compact = WHITESPACE_PATTERN.sub("", query or "")
        if not compact:
            return result

        # 安全阀 3：非劳动争议词命中 → 整条 query 不扩展（保护拒答题）
        # 整条退出而不是"跳过该词"：出现非劳动法话题时，任何扩展词都可能把
        # 拒答题推成"看起来有依据"的答案，风险大于收益
        hit_negatives = [word for word in self.negative_keywords if word in compact]
        if hit_negatives:
            result.skipped.append(f"负向规则命中：{'、'.join(hit_negatives)}")
            return result

        # 安全阀 1：易混术语对——query 已出现对中任一标准术语，则整对词条都不追加
        # 例：query 说"经济补偿"，就不能再追加"赔偿金"，否则 BM25 会把两条
        # 适用条件完全不同的条文一起召回，反而污染上下文
        blocked: set[str] = set()
        for pair in self.table.confusable_pairs:
            if any(term in compact for term in pair):
                blocked.update(pair)

        for group in self.table.groups:
            matched = [
                alternative
                for alternative in group.alternatives
                if self._matches(compact, alternative)
            ]
            if not matched:
                continue
            for term in group.standard:
                if term in blocked:
                    result.skipped.append(f"易混术语对保护，跳过「{term}」")
                    continue
                if term in compact:
                    result.skipped.append(f"query 已含法条用语，跳过「{term}」")
                    continue
                # 去重：多组可能追加同一个法条用语，只留一次
                if term in result.phrases:
                    continue
                result.phrases.append(term)
                # matched[0] 只记首个命中的替代表达：诊断需要的是"这组为什么被触发"
                result.matches.append((group.id, matched[0], term))
        return result

    @staticmethod
    def _matches(compact_query: str, alternative: str) -> bool:
        """匹配替代表达：ASCII 术语整词匹配（N 不在 2N 内部命中），中文子串匹配。

        参数：compact_query —— 已去掉空白的 query；alternative —— 替代表达。
        返回：是否命中。

        ASCII 用 lookaround 卡词边界而不用 \\b：\\b 在 "2N" 这种"数字+字母"串上
        会把边界判在 2 与 N 之间，导致 N 组被误触发；用 [0-9A-Za-z] 双向排他才是
        真正的"整词"。
        """
        if ASCII_TERM_PATTERN.match(alternative):
            pattern = re.compile(
                rf"(?<![0-9A-Za-z]){re.escape(alternative)}(?![0-9A-Za-z])",
                re.IGNORECASE,
            )
            return pattern.search(compact_query) is not None
        # 中文不设边界：中文无天然分词边界，子串命中即可（术语表本身已避免歧义词）
        return alternative in compact_query
