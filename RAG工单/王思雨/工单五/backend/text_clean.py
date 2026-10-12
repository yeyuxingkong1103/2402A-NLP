# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：文本清洗。招股说明书是带斜向水印的版本，水印不在文本层，MinerU 读页面图像时必然把它读进单元格。
#           本模块提供"水印清洗"与"污染判定"两个能力，供解析、分块、重排各环节统一调用。
import difflib                                                                  # 用相似度判定 OCR 变体的水印残渣
import html as _html                                                            # 还原 &amp; 等 HTML 实体
import re                                                                       # 正则：解析标签、剔除噪声
from functools import lru_cache                                                 # 同一词只判定一次，避免逐格重复算相似度

# 背景（实测结论，来自工单三的比对）：水印内容为"八维教育 / 人工智能刘敏 / rengongzhineng liumin"斜向重复。
# MinerU 的 -m txt 走文本层也一样出现水印，所以只能在结构化阶段后处理清洗。
# 下面两张词表由实测比对得出：拉丁残片都是 rengongzhineng / liumin 的子串；"八维""数育"在干净文本层中出现 0 次。
_NOISE = re.compile(r"(?<=[一-鿿])[a-z]{2,9}|[a-z][a-z0-9]{1,8}(?=[一-鿿])|(?<=[0-9])[a-z]{2,9}")  # 紧贴中文或数字的拉丁残渣
_KEEP = {"ms", "mbps", "qps", "mp4", "ios", "ps", "mb", "kb", "gb", "hz", "khz",              # 合法单位与技术词白名单
         "mhz", "ghz", "dpi", "ppi", "min", "mm", "cm", "km", "kg", "kw",                       # 续：长度/功率单位
         "sdk", "api", "it", "ai", "usb", "led", "lcd", "ic", "pcb", "fae", "erp", "oa",        # 续：行业缩写
         "mouser", "digi", "key", "rs", "tti", "fujitsu", "microchip", "freescale", "farnell"}  # 续：正文中确有的英文专名
_WM_LATIN = "rengongzhinengliumin"                                              # 水印拉丁文基准串
_WM_ANY = re.compile(r"八维教育|维教育|维教|八维|数育|刘敏")                      # 实测：这些词在 PDF 干净文本层出现 0 次，可任意位置剔除
_WM_EDGE = re.compile(r"^(?:人工智能)+|(?:人工智能)+$")                           # "人工智能"在正文合法出现，故只剔首尾
_WM_SYMBOL = re.compile(r"[◆◇■□●○▲△★☆※]")                                      # 水印与项目符号残留的装饰字符
_WM_TOKENS = ("八维", "维教育", "rengongzhineng", "liumin")                       # 污染判定的关键词表（用于给块降权，不做删除）


@lru_cache(maxsize=8192)                                                        # 缓存判定结果，避免重复计算相似度
def _is_wm_latin(tok: str) -> bool:                                             # 判断拉丁串是否为水印
    """水印串本身、其子串以及 OCR 变体都算水印；engineering、ionic 这类整词相似度低，不会被误删。"""
    if tok in _KEEP:                                                            # 白名单内的合法词一律保留
        return False                                                            # 不是水印
    return tok in _WM_LATIN or difflib.SequenceMatcher(None, tok, _WM_LATIN).ratio() > 0.6  # 子串命中或相似度超阈值


@lru_cache(maxsize=8192)                                                        # 同上：同一个 token 只判一次
def _is_keep(tok: str) -> bool:                                                 # 判断是否白名单词
    """抽出成函数以便 lru_cache 生效，避免在正则回调里做集合查找以外的开销。"""
    return tok in _KEEP                                                         # 命中白名单返回 True


def clean_text(text: str) -> str:                                               # 通用文本清洗（对正文安全，对表格必需）
    """剔水印、去标签、压空白。正文本就干净，清洗后基本不变；表格块则能显著恢复可检索性。"""
    s = _html.unescape(re.sub(r"<[^>]+>", " ", text or ""))                     # 去 HTML 标签并还原实体
    s = _NOISE.sub(lambda m: m.group(0) if _is_keep(m.group(0)) else "", s)     # 剔紧贴中文/数字的短噪声
    s = re.sub(r"[a-z][a-z0-9]*",                                              # 逐个小写拉丁词处理
               lambda m: "" if _is_wm_latin(m.group(0)) else m.group(0), s)     # 水印词删掉，合法词保留
    s = _WM_EDGE.sub("", _WM_ANY.sub("", s))                                    # 先剔任意位置的中文水印，再剔首尾"人工智能"
    s = _WM_SYMBOL.sub(" ", s)                                                  # 剔除装饰性符号，避免它们被当成正文
    return re.sub(r"[ \t　]+", " ", s).strip()                              # 压空白（保留换行，表格行结构要留住）


def is_noisy(text: str) -> bool:                                                # 水印污染判定
    """命中任一水印词即认为该块被污染；被污染的表格块数字与表头的对应关系不可信，检索时应降权。"""
    low = (text or "").lower()                                                  # 统一小写便于比较
    return any(w.lower() in low for w in _WM_TOKENS)                            # 任一词命中即判污染
