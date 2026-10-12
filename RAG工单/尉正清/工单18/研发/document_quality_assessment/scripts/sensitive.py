# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""敏感信息检测：手机号 / 邮箱 / 身份证 /（银行卡默认关）

## 工单强调的那句话

「关键改进是从『只展示统计数字』变成『**展示待审核列表**』，
每条匹配记录都附带**上下文**。」

所以这个模块的核心不是「找出多少条」，而是**每条都带足够的判断依据**：
文件、位置、命中内容、前后各 40 个字符的上下文。
操作员扫一眼就知道是不是误报（比如专利里的 `1.5至2.5巴` 不会被误判成手机号，
但如果误判了，上下文能让他立刻看出来）。

## 银行卡为什么默认关

工单原文：「银行卡检测默认关闭（**误报率太高**）」。
16~19 位连续数字在专利、年报里到处都是（申请号、专利号、统计表格），
开了会淹掉真正有用的信号。配置项留着，需要时打开。

## 正则里为什么要 `(?<!\\d)` 和 `(?!\\d)`

不加边界的话，`CN100342976C` 里的 `100342976` 会被当成手机号的一部分，
一串 18 位的专利号会被当成身份证。前后加数字边界能挡掉大部分这类误报。

## 光有数字边界还不够：还要挡「大数字的一截」

数字边界挡不住**用逗号千分位写的金额**。实测平安年报里两条手机号误报：

    本年提取数18921754811,6892,734      ← 其实是 1,892,175,481.1 被截出 11 位
    应收账款21,19362019822,011          ← 同类

注意这两条的前后**不是数字**（是「数」和逗号），所以 `(?<!\\d)`/`(?!\\d)` 放行。
补一条判据：命中的数字串前后**紧挨着逗号或小数点**，就说明它只是一个更大数字的
片段，不是独立号码 —— 见 `_numeric_context_ok`。

代价是「手机号后紧跟半角逗号」这种写法会被漏掉（`电话13812345678,邮箱…`）。
权衡下来值得：知识库语料里连续 11 位数字绝大多数是财务数字，
而中文文档里手机号后面跟的通常是全角逗号或空格，不受影响。
"""
import re

# 编译后的正则在进程内复用
_CACHE = {}

# 数字类命中（手机号/身份证/银行卡）前后紧邻这些字符 → 判定为「长数字的一截」
_NUM_EDGE = ",.%"


def _pat(name, pattern):
    if name not in _CACHE:
        _CACHE[name] = re.compile(pattern)
    return _CACHE[name]


# 纯数字的规则要再过一道上下文判断；邮箱不含纯数字串，不需要
_NUMERIC = {"phone", "id_card", "bank_card"}


def _numeric_context_ok(text, s, e):
    """数字类命中的附加判据：不是「大数字的一截」。

    前后紧挨着逗号/小数点/百分号，说明它属于 1,892,175,481.1 这种写法，
    见模块开头举的两条实测误报。这一条不额外读文件、不调模型，纯看字符。
    """
    if s > 0 and text[s - 1] in _NUM_EDGE:
        return False
    if e < len(text) and text[e] in _NUM_EDGE:
        return False
    return True


def scan_text(text, cfg, source=""):
    """扫一段文本，返回命中列表。

    每条形如：
      {type, value, start, context, source}
    """
    c = cfg["sensitive"]
    ctx_n = c["context_chars"]
    hits = []
    for kind in ("phone", "email", "id_card", "bank_card"):
        rule = c.get(kind, {})
        if not rule.get("enabled"):
            continue
        rx = _pat(kind, rule["pattern"])
        n = 0
        for m in rx.finditer(text):
            if n >= c["max_per_file"]:
                break
            s, e = m.start(), m.end()
            if kind in _NUMERIC and not _numeric_context_ok(text, s, e):
                continue
            n += 1
            lo, hi = max(0, s - ctx_n), min(len(text), e + ctx_n)
            hits.append({
                "type": kind,
                "value": m.group(0),
                "start": s,
                "context": re.sub(r"\s+", " ", text[lo:hi]).strip(),
                "source": source,
            })
    return hits


def file_alerts(hits, cfg):
    """命中太多的文件单独标红 —— 多半是通讯录/名单，不是技术文档。"""
    if len(hits) < cfg["sensitive"]["file_alert_count"]:
        return False
    return True


def summarize(all_hits, cfg):
    """按类型汇总，并给出「待审核列表」需要的字段。"""
    from collections import Counter
    by_type = Counter(h["type"] for h in all_hits)
    by_file = Counter(h["source"] for h in all_hits)
    alert_files = [f for f, n in by_file.items()
                   if n >= cfg["sensitive"]["file_alert_count"]]
    return {
        "total_hits": len(all_hits),
        "by_type": dict(by_type),
        "files_with_hits": len(by_file),
        # 待审核列表按「命中集中在少数文件」排序 —— 先看可疑的
        "top_files": [{"path": f, "hits": n} for f, n in by_file.most_common(20)],
        "alert_files": alert_files,
        "enabled_types": [k for k in ("phone", "email", "id_card", "bank_card")
                          if cfg["sensitive"].get(k, {}).get("enabled")],
    }
