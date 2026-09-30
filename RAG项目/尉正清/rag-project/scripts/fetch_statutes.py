# -*- coding: utf-8 -*-
"""从国家法律法规数据库镜像（lawtext/laws）提取现行法条，按「条」切分入库。

    git clone --depth 1 https://github.com/lawtext/laws /tmp/laws
    python -m scripts.fetch_statutes --src /tmp/laws/content

筛选策略：
  1. 只取「有效 / 尚未生效」状态，已废止与已修改的历史版本一律剔除，
     避免模型引用失效条文。
  2. 按 PRIORITY 顺序整部纳入，累计到 TARGET 条即止 —— 不拦腰截断某部法律，
     否则用户问到后半部就检索不到。
  3. 同名法规保留施行日期最新的一版。

数据源为国家法律法规数据库（flk.npc.gov.cn）的转换文本，属公有领域。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.metadata_service import CN_NUMERAL, CN_SUFFIX
from scripts.dataset_io import did, write_jsonl

DEFAULT_SRC = "/tmp/laws/content"
OUT_DIR = "/root/rag-project/data/lawyer"
TARGET = 4000

# 按日常法律咨询的出现频率排序；同一条文在整部法律内保持完整
PRIORITY = [
    "中华人民共和国民法典",
    "中华人民共和国刑法",
    "中华人民共和国民事诉讼法",
    "中华人民共和国刑事诉讼法",
    "中华人民共和国公司法",
    "中华人民共和国劳动法",
    "中华人民共和国劳动合同法",
    "中华人民共和国劳动争议调解仲裁法",
    "中华人民共和国治安管理处罚法",
    "中华人民共和国道路交通安全法",
    "中华人民共和国行政处罚法",
    "中华人民共和国行政诉讼法",
    "中华人民共和国消费者权益保护法",
    "中华人民共和国宪法",
    "最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释（一）",
    "最高人民法院关于适用《中华人民共和国民法典》有关担保制度的解释",
    "最高人民法院关于适用《中华人民共和国民法典》合同编通则若干问题的解释",
    "最高人民法院关于适用《中华人民共和国民法典》继承编的解释（一）",
    "最高人民法院关于民事诉讼证据的若干规定",
    "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）",
    "中华人民共和国社会保险法",
    "中华人民共和国合伙企业法",
    "中华人民共和国企业破产法",
    "中华人民共和国保险法",
    "中华人民共和国票据法",
    "中华人民共和国仲裁法",
    "中华人民共和国人民调解法",
    "中华人民共和国著作权法",
    "中华人民共和国商标法",
    "中华人民共和国专利法",
]

SKIP_DIRS = {"en", "about", "appendix"}
FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
_ART = r"第[" + CN_NUMERAL + r"]+条(?:之[" + CN_SUFFIX + r"]+)?"
ART_RE = re.compile(
    r"^\s*[-*]?\s*\*\*(" + _ART + r")\*\*\s*(.*?)"
    r"(?=\n\s*[-*]?\s*\*\*" + _ART + r"\*\*|\n#{2,6}\s|\Z)",
    re.S | re.M)
# 标题层级各法不一：《民法典》用 #### / #####，多数法律用 ## / ###，
# 因此匹配 2~6 级标题，否则会漏掉整部民法典的章节
CHAP_RE = re.compile(
    r"^#{2,6}\s*(第[" + CN_NUMERAL + r"]+[章节][^\n]*)", re.M)


def parse(path):
    text = open(path, encoding="utf-8").read()
    m = FM_RE.match(text)
    fm, body = {}, text
    if m:
        body = text[m.end():]
        for line in m.group(1).split("\n"):
            if ":" in line and not line.startswith(" "):
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip().strip("'\"")
    return fm, body


def clean(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"^[\s\-*]+", "", s, flags=re.M)
    return re.sub(r"[ \t　]+", " ", s).strip()


def worth_keeping(text):
    """滤掉没有独立检索价值的条文。"""
    if len(text) < 15:
        return False
    # 「本法自公布之日起施行」这类施行日期条款
    if re.match(r"^本法(自|所称|下列)", text) and len(text) < 40:
        return False
    return True


def load_laws(src):
    """扫描目录，返回 {标题: (施行日期, frontmatter, 正文)}，同名取最新版。"""
    best = {}
    for dp, dns, fns in os.walk(src):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in fns:
            if not fn.endswith(".md") or fn.startswith("_"):
                continue
            fm, body = parse(os.path.join(dp, fn))
            title = fm.get("title", "").strip()
            if not title or fm.get("status") not in ("有效", "尚未生效"):
                continue
            eff = fm.get("effective_date", "") or fm.get("publication_date", "")
            if title not in best or eff > best[title][0]:
                best[title] = (eff, fm, body)
    return best


def resolve(title, laws):
    """按优先级条目找对应法规；宪法取现行修正文本。"""
    if title == "中华人民共和国宪法":
        for t in laws:
            if t.startswith("中华人民共和国宪法（") and "修正文本" in t:
                return t
        return None
    if title in laws:
        return title
    return None


def build(laws, target=TARGET):
    recs, used, total = [], [], 0
    for want in PRIORITY:
        real = resolve(want, laws)
        if real is None:
            print("  [跳过] 未找到: %s" % want)
            continue
        eff, fm, body = laws[real]
        chapters = [(m.start(), clean(m.group(1))) for m in CHAP_RE.finditer(body)]

        def chapter_of(pos):
            cur = ""
            for s, name in chapters:
                if s <= pos:
                    cur = name
                else:
                    break
            return cur

        n_before = len(recs)
        for m in ART_RE.finditer(body):
            content = clean(m.group(2))
            if not worth_keeping(content):
                continue
            art_no = m.group(1)
            chap = chapter_of(m.start())
            head = "《%s》%s" % (real, art_no)
            recs.append({
                "doc_id": did("statute", real, art_no),
                "role_id": "lawyer",
                "source": "legal_statutes.jsonl",
                "doc_type": "statute",
                "embed_text": ("%s %s" % (head, content))[:800],
                "display_text": "【现行法条】%s%s\n%s" % (
                    head, ("（%s）" % chap) if chap else "", content),
                "meta": {"law": real, "article": art_no, "chapter": chap,
                         "effective_date": eff, "status": fm.get("status", "")},
            })
        added = len(recs) - n_before
        total += added
        used.append((real, added))
        print("  %5d 条  %s" % (added, real[:56]))
        if total >= target:
            print("  —— 已达目标 %d 条，停止纳入 ——" % target)
            break
    return recs, used


def main():
    src = DEFAULT_SRC
    if "--src" in sys.argv:
        src = sys.argv[sys.argv.index("--src") + 1]
    if not os.path.isdir(src):
        print("数据目录不存在: %s\n请先执行:\n"
              "  git clone --depth 1 https://github.com/lawtext/laws /tmp/laws" % src)
        return 1

    print("扫描 %s ..." % src)
    laws = load_laws(src)
    print("可用现行法规 %d 部\n" % len(laws))

    print("按优先级纳入:")
    recs, used = build(laws)
    if not recs:
        print("未提取到任何法条")
        return 1

    # 同名同条去重
    seen, uniq = set(), []
    for r in recs:
        if r["doc_id"] in seen:
            continue
        seen.add(r["doc_id"])
        uniq.append(r)

    write_jsonl(os.path.join(OUT_DIR, "legal_statutes.jsonl"), uniq)
    print("\n合计 %d 条，覆盖 %d 部法规" % (len(uniq), len(used)))
    length = [len(r["display_text"]) for r in uniq]
    print("条文长度: 平均 %d 字, 最长 %d 字" % (sum(length) // len(length), max(length)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
