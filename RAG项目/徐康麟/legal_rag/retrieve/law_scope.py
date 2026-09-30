"""法名定向召回：问题里点名了哪部法，就额外去**那部法的文件**里捞一遍候选。

为什么需要（评测 v1/v2 的结论）：L07 / L13 / L23 / L24 四道未过题的共同根因不是重排，
而是**该法条压根没进候选池**——向量 top-10 + 关键词 top-10 里全是相邻的司法解释与
民法典条文（比如问"什么行为构成商标侵权"，捞回来的却是反不正当竞争法解释），
重排再好也救不了缺失的候选。

而法律问题的**法名往往就在问句里**（"商标侵权""著作权保护期""正当防卫"），
这是比向量相似度可靠得多的强线索：用它做一次 ``source in [该法的文件…]`` 的定向召回，
把该法的真条文**加进**候选池（**只加不删**，不做硬过滤 —— 一问多法时硬过滤会丢召回）。

判据仍是词面的（法名/触发词命中即算），因此宁宽勿严：多召回的噪声交给重排去压。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

#: 法名 → 触发词。**人工维护**：新场景就往上加（与 ``engine._LEGAL_SIGNAL_WORDS`` 同理，
#: 区别是这里只认"哪部法"，用于定向召回）。触发词按长度降序匹配，避免短词抢先。
LAW_TRIGGERS: dict[str, tuple[str, ...]] = {
    "民法典": ("民法典", "善意取得", "不当得利", "无因管理", "诉讼时效", "表见代理"),
    "刑法": ("刑法", "正当防卫", "紧急避险", "自首", "立功", "缓刑", "量刑", "犯罪构成",
             "故意伤害", "盗窃罪", "诈骗罪", "交通肇事罪"),
    "道路交通安全法": ("道路交通安全法", "醉驾", "酒驾", "闯红灯", "交通事故认定",
                        "逃逸", "吊销驾驶证", "机动车", "交强险"),
    "消费者权益保护法": ("消费者权益保护法", "消保法", "假货", "退一赔三", "三倍赔偿",
                         "欺诈消费者", "七天无理由", "消费者"),
    "劳动合同法": ("劳动合同法", "劳动合同", "试用期", "经济补偿", "竞业限制",
                   "无固定期限", "解除劳动合同"),
    "劳动争议调解仲裁法": ("劳动争议调解仲裁法", "劳动仲裁", "仲裁时效", "劳动争议"),
    "公司法": ("公司法", "股东", "出资", "董事会", "监事", "股权转让", "注册资本"),
    "民事诉讼法": ("民事诉讼法", "民诉", "举证责任", "管辖", "一审", "二审", "再审",
                   "执行程序", "诉讼保全"),
    "刑事诉讼法": ("刑事诉讼法", "刑诉", "取保候审", "羁押", "侦查阶段", "不起诉"),
    "商标法": ("商标法", "商标侵权", "注册商标", "商标注册", "商标专用权", "近似商标"),
    "专利法": ("专利法", "专利权", "专利侵权", "发明专利", "实用新型", "外观设计"),
    "著作权法": ("著作权法", "著作权", "版权", "作品保护期", "信息网络传播权", "署名权"),
    "个人信息保护法": ("个人信息保护法", "个人信息", "隐私政策", "敏感个人信息", "人脸识别"),
    "证券法": ("证券法", "内幕交易", "信息披露", "操纵市场", "上市公司", "证券发行"),
    "治安管理处罚法": ("治安管理处罚法", "治安处罚", "行政拘留", "寻衅滋事", "打架斗殴"),
    "行政处罚法": ("行政处罚法", "行政处罚", "罚款", "听证", "执法程序"),
    "产品质量法": ("产品质量法", "产品缺陷", "产品质量", "生产者责任", "三包"),
    "国家赔偿法": ("国家赔偿法", "国家赔偿", "冤假错案", "刑事赔偿", "行政赔偿"),
    "企业破产法": ("企业破产法", "破产", "清算", "重整", "清偿顺序"),
    "社会保险法": ("社会保险法", "社保", "养老保险", "医疗保险", "失业保险"),
    "工伤保险条例": ("工伤保险条例", "工伤", "工伤认定", "工伤赔偿", "职业病"),
    "妇女权益保障法": ("妇女权益保障法", "女职工", "孕期", "产假", "性骚扰"),
    "未成年人保护法": ("未成年人保护法", "未成年人", "未成年", "打赏主播", "校园欺凌"),
    "宪法": ("宪法", "基本权利", "公民权利", "国家机构"),
    "监察法": ("监察法", "监察机关", "留置", "政务处分"),
    "婚姻家庭编": ("婚姻家庭编", "结婚", "离婚", "抚养权", "夫妻共同财产", "彩礼"),
    "继承编": ("继承编", "继承", "遗嘱", "遗产", "法定继承"),
    # 下面是**主题族**（不是法典名，但语料里成体系、问句里常直接点名），
    # 匹配规则与法名一致：语料文件名里含这个名字就算它的人。
    "民间借贷": ("民间借贷", "借款利率", "LPR", "利息上限", "借条利率"),
}

#: 书名号里的名字（《…》）与语料文件名都会出现这两种写法
_BRACKET_TRANS = str.maketrans({"〈": "《", "〉": "》"})
_BOOK_TITLE_RE = re.compile(r"《([^》]{2,40})》")
_SOURCE_SUFFIX_RE = re.compile(r"(__[0-9a-f]{6,})?\.(md|txt|json|pdf)$", re.IGNORECASE)
#: 语料文件名常见前缀（去掉后才能与短名对上）
_LAW_PREFIXES = ("中华人民共和国", "最高人民法院关于", "最高人民法院", "关于")


def normalize_law_name(name: str) -> str:
    """统一书名号并去空白。"""
    return (name or "").translate(_BRACKET_TRANS).strip()


def law_name_from_source(source: str) -> str:
    """语料文件名 → 法名（去目录、去 ``__hash``、去扩展名）。"""
    text = str(source or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return _SOURCE_SUFFIX_RE.sub("", text)


#: 配套法规后缀（紧跟法典本体的第二位）
CANON_SUFFIXES = ("实施条例", "实施细则", "实施办法", "实施规定")


#: 司法解释/规范性文件的文件名前缀（**注意别拿 ``_LAW_PREFIXES`` 当这个用**：后者含
#: "中华人民共和国"，那恰恰是**法典本体**的前缀，用错会把《商标法》本体判成司法解释）。
_COURT_PREFIXES = ("最高人民法院", "最高人民检察院", "关于")


def canon_rank(name: str, key: str) -> int:
    """同一法名族内的"正统程度"：**0 = 法典本体**，1 = 配套法规，2 = 司法解释/修改决定等。

    定向召回按这个顺序取候选，所以法典本体必须排最前 —— 讲"什么行为构成商标侵权"的是
    《商标法》第五十七条，而不是《最高人民法院关于商标法修改决定施行后…的解释》。
    """
    cleaned = law_name_from_source(name)
    if cleaned.startswith(_COURT_PREFIXES):
        return 2                         # 最高人民法院…/关于… ⇒ 司法解释
    bare = cleaned
    for prefix in _LAW_PREFIXES:
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
    if bare in (key, f"中华人民共和国{key}") or cleaned in (key, f"中华人民共和国{key}"):
        return 0                         # 法典本体
    if any(suffix in cleaned for suffix in CANON_SUFFIXES):
        return 1                         # 实施条例/细则
    if any(word in cleaned for word in ("解释", "决定", "批复", "规定", "纪要", "意见")):
        return 2                         # 司法解释类
    return 1                             # 其余同名文件（如"要点"）当作配套


def canonical_law_name(name: str) -> str:
    """法名归一化到 :data:`LAW_TRIGGERS` 的键（对不上就返回去掉前缀的全名）。"""
    bare = normalize_law_name(name)
    if bare in LAW_TRIGGERS:
        return bare
    for prefix in _LAW_PREFIXES:
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
    if bare in LAW_TRIGGERS:
        return bare
    # 键包含关系：《中华人民共和国道路交通安全法实施条例》→ 道路交通安全法
    for key in sorted(LAW_TRIGGERS, key=len, reverse=True):
        if key in bare:
            return key
    return bare


def detect_law_mentions(question: str, limit: int = 3) -> list[str]:
    """问题里提到了哪几部法（返回 :data:`LAW_TRIGGERS` 的键），按确定度排序。

    先认显式书名号（用户直接写了法名），再认触发词；两类都命中的排前面。
    找不到就返回空列表（此时不做定向召回，行为与改动前完全一致）。
    """
    text = str(question or "")
    if not text.strip():
        return []

    explicit: list[str] = []
    for match in _BOOK_TITLE_RE.finditer(text):
        key = canonical_law_name(match.group(1))
        if key in LAW_TRIGGERS and key not in explicit:
            explicit.append(key)

    triggered: list[str] = []
    # 长触发词优先（"劳动争议调解仲裁法" 不该被 "劳动争议" 抢先）
    pairs = sorted(
        ((key, word) for key, words in LAW_TRIGGERS.items() for word in words),
        key=lambda item: len(item[1]), reverse=True,
    )
    for key, word in pairs:
        if word in text and key not in triggered:
            triggered.append(key)

    ordered = explicit + [key for key in triggered if key not in explicit]
    return ordered[:limit]


def build_law_source_map(paths: Iterable[str | Path]) -> dict[str, list[str]]:
    """扫语料文件名，建「法名 → 该法的文件名列表」映射（用 basename，跨机器可移植）。

    列表**按"法典本体优先"排序**（见 :func:`canon_rank`）：同名族里既有《中华人民共和国商标法》
    又有《最高人民法院关于商标法修改决定施行后…的解释》时，法典本体必须排前面 ——
    定向召回是按这个顺序取 top_k 的，排错了就会把名额全给司法解释（真机踩过：
    商标法/公司法两题的 source 列表第一项都是司法解释，结果法典本体只挤进一两条）。
    """
    mapping: dict[str, list[str]] = {}
    for raw in paths:
        path = Path(raw)
        if path.suffix.lower() not in (".md", ".txt", ".json", ".pdf"):
            continue
        name = path.name
        full = law_name_from_source(name)
        for key in LAW_TRIGGERS:
            if key in full:
                bucket = mapping.setdefault(key, [])
                if name not in bucket:
                    bucket.append(name)
    for key, names in mapping.items():
        names.sort(key=lambda item: (canon_rank(item, key), item))
    return mapping


@dataclass
class LawScope:
    """法名 → 语料文件名 的运行时索引（很小，直接常驻内存）。"""

    mapping: dict[str, list[str]] = field(default_factory=dict)

    # ---------- 构造 ----------
    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> "LawScope":
        return cls(build_law_source_map(paths))

    @classmethod
    def load(cls, path: str | Path) -> "LawScope":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        mapping = {str(k): [str(x) for x in v] for k, v in dict(payload or {}).items()}
        return cls(mapping)

    @classmethod
    def load_or_build(cls, index_dir: str | Path, corpus_dir: str | Path) -> "LawScope":
        """优先读 ``<index_dir>/law_sources.json``；没有就扫语料目录（只读文件名，很快）。

        两者都拿不到 ⇒ 返回空索引（定向召回**自动不生效**，其余行为一律不变）。
        """
        cache = Path(index_dir) / "law_sources.json"
        if cache.is_file():
            try:
                scope = cls.load(cache)
                if scope.mapping:
                    return scope
            except (OSError, json.JSONDecodeError, TypeError) as exc:  # noqa: BLE001
                logger.warning("法名索引读失败（忽略，改扫语料）：path=%s 原因=%s: %s",
                               cache, type(exc).__name__, exc)
        root = Path(corpus_dir)
        if root.is_dir():
            scope = cls.from_paths(root.rglob("*"))
            if scope.mapping:
                logger.info("法名定向召回索引已就绪：%d 部法（来源=%s）", len(scope.mapping), root)
            return scope
        logger.info("没有法名索引（既无 %s 也无语料目录 %s）：本轮不做定向召回", cache, root)
        return cls({})

    def dump(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.mapping, ensure_ascii=False, indent=2, sort_keys=True),
                          encoding="utf-8")
        return target

    # ---------- 查询 ----------
    def sources_for(self, question: str, limit: int = 3) -> list[str]:
        """问题点到的法在语料里的文件名（多部法取并集，**法典本体优先**）。"""
        found: list[str] = []
        for _label, sources in self.groups_for(question, limit=limit):
            for name in sources:
                if name not in found:
                    found.append(name)
        return found

    def groups_for(self, question: str, limit: int = 3) -> list[tuple[str, list[str]]]:
        """按"正统程度"把该法的文件分成 ``[("canon", [...]), ("related", [...])]``。

        分两组的理由：定向召回**分别**去这两组取 top_k。合在一起取时，司法解释常常把名额
        挤掉（真机踩过），而讲"什么行为构成商标侵权"的恰是《商标法》第五十七条。
        """
        canon: list[str] = []
        related: list[str] = []
        for law in detect_law_mentions(question, limit=limit):
            for name in self.mapping.get(law, []):
                target = canon if canon_rank(name, law) <= 1 else related
                if name not in target:
                    target.append(name)
        groups: list[tuple[str, list[str]]] = []
        if canon:
            groups.append(("canon", canon))
        if related:
            groups.append(("related", related))
        return groups

    def laws_for(self, question: str, limit: int = 3) -> list[str]:
        return [law for law in detect_law_mentions(question, limit=limit) if self.mapping.get(law)]

    def __len__(self) -> int:
        return len(self.mapping)
