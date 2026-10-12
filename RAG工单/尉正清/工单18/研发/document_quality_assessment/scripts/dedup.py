# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""重复检测：MD5 精确匹配 + SimHash 近似匹配

## 两层机制的区别（工单原文）

「MD5 用于识别完全相同的文件，SimHash 用于识别高相似度的文档
（比如不同版本的同一份文件）。**MD5 是精确匹配，SimHash 需要人工确认**，
所以输出是『待确认的版本冲突列表』。」

这个区分很重要，直接决定了报告的两种呈现：

| | 结论 | 报告里怎么放 |
|---|---|---|
| MD5 相同 | 就是同一个文件 | 「**完全重复**」分组，可直接处理 |
| SimHash 相近 | 可能是不同版本 | 「**待确认的版本冲突列表**」，每条带相似度让人判 |

## SimHash 取哪些字符：头部 + 中部 + 尾部，不是只取头部

全文算 SimHash 要分词 + 加权，1700 份文档太慢，所以只能取样。
最初取的是**头部**，理由是「同一份文档的不同版本，头部足以区分」。**这个是错的**，
实测被数据打脸：知识库里的年报、招股书、专利，**开头恰恰是最套话的部分**
（封面、免责声明、目录、格式条款）。只看头部：

    distance 4   招商证券年报  <->  国泰君安年报      ← 两家不同公司
    distance 6   招股说明书1-无水印.pdf <-> 招股说明书1.pdf  ← 同一份文档的两个版本

**真正的版本对反而排在假的对后面** —— 这种排序下，阈值调多少都分不开。
所以改成正文取样：头一半、中四分之一、尾四分之一。
中后段才装得下各自的财务数据、公司名、业务描述，区分度在那里。

## 比较是 O(n²) 的

n 份文档两两比较要 n(n-1)/2 次。1700 份就是 144 万次汉明距离计算 ——
用位运算做（xor + popcount）其实很快（实测 1700 份几秒），
但配置里还是留了 `simhash_max_files` 上限，防止有人拿几万份文档来跑。
"""
import hashlib
import re


def md5_groups(files, md5_of):
    """按 MD5 分组，返回 {md5: [路径, ...]}，只保留多于 1 个文件的组。"""
    seen = {}
    for p in files:
        try:
            h = md5_of(p)
        except OSError:
            continue
        seen.setdefault(h, []).append(str(p))
    return {h: ps for h, ps in seen.items() if len(ps) > 1}


# ---------------------------------------------------------------- SimHash

_CJK = re.compile(r"[一-鿿]+")
_WORD = re.compile(r"[A-Za-z0-9]+")


def _shingles(text, n=3):
    """切 shingle：英文数字按词，中文按 n 字滑窗。

    ## 为什么不是「按单字切」（最初就是那么写的，被实测打脸）

    单个汉字的熵太低。「的、一、是、在、有」这类字每份中文文档里都是常客，
    SimHash 累加下来，两份**毫不相干**的专利也能有 92% 的二进制位相同 ——
    阈值 5 那档捞出 19 对，人工核对发现标题对得上的只有三分之一。

    改成长度 n 的滑窗后，比的是「短语」而不是「用字偏好」：
    「带有预前处理的水泥磨」和「螺旋内刮式自清洗过滤器」在单字层面很像，
    在三字滑窗层面几乎不重叠。**不用引分词器**，滑窗就够，
    少一个依赖少一份风险。
    """
    out = []
    for run in _CJK.findall(text):
        if len(run) <= n:
            out.append(run)
        else:
            out.extend(run[i:i + n] for i in range(len(run) - n + 1))
    out.extend(_WORD.findall(text))
    return out


def sample_text(text, n):
    """从全文里取 n 个字符：头 1/2、中 1/4、尾 1/4。

    取样而不是取头部的理由见模块开头 —— 头部是文档里最套话的部分。
    短于 n 的文档整篇返回。
    """
    if n <= 0 or len(text) <= n:
        return text
    head = n // 2
    rest = n - head
    mid, tail = rest // 2, rest - rest // 2
    m0 = (len(text) - mid) // 2
    return text[:head] + "\n" + text[m0:m0 + mid] + "\n" + (text[-tail:] if tail else "")


def simhash(text, bits=64, shingle=3):
    """算 SimHash。返回整数；文本为空返回 0。"""
    tks = _shingles(text, shingle)
    if not tks:
        return 0
    weights = [0] * bits
    for tk in tks:
        h = int(hashlib.md5(tk.encode("utf-8")).hexdigest(), 16) & ((1 << bits) - 1)
        for i in range(bits):
            weights[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, w in enumerate(weights):
        if w > 0:
            out |= (1 << i)
    return out


def hamming(a, b):
    return bin(a ^ b).count("1")


def simhash_pairs_from_hashes(items, cfg):
    """用**已算好的** SimHash 两两比较。items 是 [(路径, hash), ...]。

    主流程走这个 —— hash 在解析阶段就算好存进缓存了，出报告时不该再读一遍正文。
    """
    c = cfg["dedup"]
    bits = c["simhash_bits"]
    max_files = c["simhash_max_files"]
    if len(items) > max_files:
        return None, f"文件数 {len(items)} 超过上限 {max_files}，跳过 SimHash（配置项 simhash_max_files）"
    if len(items) < 2:
        return [], ""

    thr = c["simhash_distance"]
    pairs = []
    for i in range(len(items)):
        pa, ha = items[i]
        for j in range(i + 1, len(items)):
            pb, hb = items[j]
            d = hamming(ha, hb)
            if d <= thr:
                pairs.append({"a": pa, "b": pb, "distance": d,
                              "similarity": round(1 - d / bits, 4)})
    pairs.sort(key=lambda x: x["distance"])
    return pairs, ""


def simhash_pairs(files, texts, cfg):
    """直接从文本算 SimHash 再比较（不依赖缓存）。

    files 与 texts 一一对应，用于单测和「只有文本没有缓存」的场合。
    """
    c = cfg["dedup"]
    bits = c["simhash_bits"]
    hashes = [(str(p), simhash(sample_text(t, c["simhash_sample_chars"]), bits,
                               c["simhash_shingle"]))
              for p, t in zip(files, texts) if t]
    return simhash_pairs_from_hashes(hashes, cfg)


def summarize(md5_dup, sim_pairs, total_files):
    """汇总。sim_pairs 为 None 表示跳过了。"""
    wasted = 0
    for ps in md5_dup.values():
        # 一组 n 个完全相同的文件，其中 n-1 个是多余的
        wasted += len(ps) - 1
    return {
        "total_files": total_files,
        "exact_dup_groups": len(md5_dup),
        "exact_dup_files": sum(len(v) for v in md5_dup.values()),
        "redundant_copies": wasted,
        "similar_pairs": None if sim_pairs is None else len(sim_pairs),
    }
