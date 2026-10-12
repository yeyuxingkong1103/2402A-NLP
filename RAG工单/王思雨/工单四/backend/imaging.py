# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 文件说明：★工单四核心★ 图像内容解析。多模态模型 Chinese-CLIP 判定图像语义类型，PaddleOCR 读出图内文字，
# 再按"横总线+下垂竖线"的制图规律还原层级关系（如"大客户销售部 下辖 6 个销售处"），合成可检索的图像语义块。
import re                                                                   # 文本清洗
import numpy as np                                                          # 图像矩阵与线检测
from PIL import Image                                                       # 图像读写与放大
import config as C                                                          # 集中配置

_CLIP, _PROC, _OCR = None, None, None                                       # CLIP 模型/处理器、OCR 单例

# CLIP 零样本候选标签：覆盖招股说明书里会出现的图种类，判定结果直接写进图像块的语义描述
LABELS = ["组织结构图", "流程图", "柱状图", "饼图", "折线图", "数据表格", "公司印章",
          "手写签名", "产品实物照片", "人物照片", "公司标识logo"]
# 只有结构框图才适用"横总线+下垂竖线"的层级规律；饼图/柱状图的图形元素会被误判成总线，必须排除
TREE_TYPES = ("组织结构图",)


def clip_model():                                                           # 加载中文 CLIP
    """本地加载多模态模型（工单备注要求 CLIP 或多模态大模型，此处用中文 CLIP 以支持中文提示词）。"""
    global _CLIP, _PROC                                                     # 声明单例
    if _CLIP is None:                                                       # 首次调用才加载
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor     # 延迟导入
        _CLIP = ChineseCLIPModel.from_pretrained(C.CLIP_MODEL_PATH).to(C.DEVICE).eval()  # 本地权重，不联网
        _PROC = ChineseCLIPProcessor.from_pretrained(C.CLIP_MODEL_PATH)     # 配套预处理
    return _CLIP, _PROC                                                     # 返回单例


def ocr_engine():                                                           # 加载 PaddleOCR
    """模型目录显式指向纯英文路径：用户名含中文时 Paddle 的 C++ 推理层打不开默认缓存路径。"""
    global _OCR                                                             # 声明单例
    if _OCR is None:                                                        # 首次调用才加载
        from paddleocr import PaddleOCR                                     # 延迟导入
        d = C.PADDLEOCR_MODEL_DIR                                           # 模型根目录
        _OCR = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False,      # 中文模型 + 方向分类
                         det_model_dir=f"{d}/det", rec_model_dir=f"{d}/rec", cls_model_dir=f"{d}/cls")  # 显式指定三件套
    return _OCR                                                             # 返回单例


def is_watermark(text: str) -> bool:                                        # 水印判定
    """招股说明书的斜向水印不在文本层，OCR 会把"八维教育/人工智能刘敏/rengongzhinengliumin"读成独立文本行，必须滤掉。"""
    low = (text or "").lower()                                              # 统一小写比对
    return any(w.lower() in low for w in C.WATERMARK_TOKENS)                # 命中任一词表即判定为水印


def box_ok(size: tuple) -> bool:                                            # 图像可用性判定
    """过小的图多是水印碎片、项目符号、装饰线，送进 CLIP/OCR 只会产生噪声块。"""
    return bool(size) and min(size) >= C.IMAGE_MIN_PX                       # 两边都不小于阈值


def classify(img: Image.Image) -> tuple:                                    # ★多模态语义判定
    """Chinese-CLIP 零样本分类：给出图像语义类型与置信度，作为图像块的第一行描述。"""
    import torch                                                            # 延迟导入
    m, p = clip_model()                                                     # 取单例
    with torch.no_grad():                                                   # 推理不需要梯度
        logits = m(**p(text=LABELS, images=img, return_tensors="pt", padding=True).to(C.DEVICE)).logits_per_image  # 图文相似度
        prob = logits.softmax(-1)[0]                                        # 转概率
    i = int(prob.argmax())                                                  # 取最高分标签
    return LABELS[i], round(float(prob[i]), 3)                              # 语义类型与置信度


def read_lines(img: Image.Image) -> list:                                   # 图内文字识别
    """放大后 OCR：实测 1x 会把图表里的负号读丢（"-2.0%" 读成 "2.0%"），3x 放大才能稳定读出。"""
    big = img.resize((int(img.width * C.OCR_SCALE), int(img.height * C.OCR_SCALE)), Image.LANCZOS)  # 放大重采样
    res = ocr_engine().ocr(np.array(big), cls=True)                         # 识别
    out = []                                                                # 结果：(文本, 左上x, 上y, 右下x, 下y) 按原图坐标
    for line in (res[0] or []):                                             # 逐检测框
        txt = (line[1][0] or "").strip()                                    # 文本
        conf = float(line[1][1])                                            # 置信度
        xs = [q[0] / C.OCR_SCALE for q in line[0]]                          # 还原到原图 x
        ys = [q[1] / C.OCR_SCALE for q in line[0]]                          # 还原到原图 y
        if txt and conf >= C.OCR_MIN_CONF and not is_watermark(txt) and len(txt) > 1:  # 滤水印/低置信/单字
            out.append((txt, min(xs), min(ys), max(xs), max(ys)))           # 收下
    return out                                                              # 返回干净文本行


_VALUE = re.compile(r"^[-−—]?\d[\d,，]*(\.\d+)?%?$")                          # 纯数值 token：含负数、千分位、百分号


def num_pairs(lines: list, width: int) -> list:                             # ★图表"标签: 数值"配对
    """把条形图/柱状图的标签与数值配成对，避免 OCR 阅读顺序把两个面板的数字错配。

    实测踩坑：IC 市场增长图上，"汽车"的增长率 14.0% 与左边饼图的"消费 26%"在阅读顺序里相邻，
    模型据此答成"汽车增长率 26%"——**数字张冠李戴，且关键词判分会误判为正确**。
    条形图的画法固定：标签在左、数值在条形右端，二者水平间隔大而垂直几乎同行；
    饼图的标签与数值则紧挨着（水平投影重叠）。据此只取"水平间隔 > 40px 且垂直差 < 45px"的配对，
    并按垂直距离贪心一对一分配，保证每个数值只配一个标签。
    """
    vals = [l for l in lines if _VALUE.match(l[0].strip())]                  # 数值行
    # 标签必须不含数字：OCR 常把饼图里的若干小字粘成一坨（如"工控13710248"、"2% 1%"），
    # 这类混合串既不是标签也不是数值，参与配对会把正经标签挤掉。
    # 标签还须含中日韩字符：招股说明书里的图表标签都是中文，纯拉丁串基本是水印被 OCR 打碎的残渣
    # （实测组织结构图上配出过 "ongzhi 100%" 这种无意义对）。
    labs = [l for l in lines if not _VALUE.match(l[0].strip()) and not re.search(r"\d", l[0])
            and re.search(r"[一-鿿]", l[0])]                                 # 纯中文标签行
    ul, uv, out = set(), set(), []                                          # 已占用的标签/数值，与结果
    # ── 模式二（先做）：标签与数值**水平投影相交**，即饼图式紧密排布，以及负值条形图
    #    （负值画在轴线左侧，数值会压到标签列上，实测 IC卡与"-2.0%"就是这种）。
    #    只配带百分号的数值，避免把饼图的绝对金额（2515、1541）也当成增长率。
    for vi, v in enumerate(vals):                                            # 逐个数值
        if not v[0].strip().endswith("%"): continue                          # 只认百分比
        cy = (v[2] + v[4]) / 2                                               # 垂直中心
        hit = [(abs((l[2] + l[4]) / 2 - cy), li) for li, l in enumerate(labs)  # 水平相交的标签
               if v[1] < l[3] and v[3] > l[1]]
        if not hit: continue                                                 # 没有相交标签
        dy, li = min(hit)                                                    # 取垂直最近者
        if dy < 40:                                                          # 同一视觉组
            uv.add(vi); ul.add(li); out.append(f"{labs[li][0].strip()} {v[0].strip()}")  # 配对成功
    # ── 模式一：条形图布局——标签在左、数值在条形右端，水平间隔大而几乎同行。
    #    先做模式二的意义：饼图的标签（如"计算机""消费"）一旦被自己的数值占用，
    #    就不会再跨越大半张图去抢条形图的数值（实测曾配出"计算机 -2.0%""消费 14.0%"）。
    cand = []                                                                # 候选配对
    for vi, v in enumerate(vals):                                            # 逐个数值
        if vi in uv: continue                                                # 已在模式二配对
        cy = (v[2] + v[4]) / 2                                               # 数值的垂直中心
        for li, l in enumerate(labs):                                        # 逐个标签
            if li in ul: continue                                            # 已被占用
            gap = v[1] - l[3]                                                # 标签右边界到数值左边界的水平间隔
            dy = abs((l[2] + l[4]) / 2 - cy)                                 # 垂直距离
            # 垂直阈值 40px 是实测出来的：正确配对最大到 32.5px（"汽车"的标签画在数值下方一行），
            # 错误配对（"-2.0%"配给"计算机"）有 147px。40 正好把两者分开。
            if gap > 40 and dy < 40: cand.append((dy + gap * 0.1, vi, li))    # 代价同时看垂直与水平
    cand.sort()                                                             # 代价小者优先
    for _, vi, li in cand:                                                  # 贪心一对一分配
        if vi in uv or li in ul: continue                                   # 已被占用则跳过
        uv.add(vi); ul.add(li)                                              # 占用
        out.append(f"{labs[li][0].strip()} {vals[vi][0].strip()}")           # 组成"标签 数值"
    return out[:16]                                                         # 数量上限


def relations(dark: np.ndarray, lines: list) -> list:                       # ★图内层级还原
    """按制图规律还原"谁下辖谁"。

    组织结构图/流程图的画法固定：父块下方引一条**横总线**，总线上挂多条**下垂竖线**接到各个子块。
    于是只要在图中找出"挂着 ≥2 条竖线的横线"，其横跨范围就是一组兄弟节点，正上方最近的块就是它们的父节点。
    实测这条规律能正确还原"销售部→4 个部门""大客户销售部→6 个销售处"。
    """
    H, W = dark.shape                                                       # 尺寸
    def up_bar(x, y, n=18):                                                 # 该点上方是否有竖线（用于排除"框边线"）
        return bool(dark[max(0, y - n):y - 1, max(0, x - 1):x + 2].any())    # 取一窄条判断
    def down_run(x, y, n=18):                                               # 该点下方竖线长度（下垂线）
        return int(dark[y + 2:y + n, max(0, x - 1):x + 2].sum())            # 计数
    cands = []                                                              # 候选总线
    for y in range(2, H - 20):                                              # 逐行扫描
        s = None                                                            # 当前暗线段起点
        for x in range(W):                                                  # 逐列
            if dark[y, x] and s is None: s = x                              # 段开始
            elif not dark[y, x] and s is not None:                          # 段结束
                a, b = s, x - 1                                             # 段范围
                s = None                                                    # 复位
                if b - a <= W * 0.05: continue                              # 太短，不是总线
                if up_bar(a, y) and up_bar(b, y): continue                  # 两端都有上延竖线 = 方框边线，排除
                drops, px = [], -99                                         # 下垂竖线位置
                for xx in range(a, b + 1):                                  # 沿总线找垂线
                    if down_run(xx, y) > 10 and xx - px > 30: drops.append(xx); px = xx  # 间隔>30px 才算独立分叉
                if len(drops) >= 2: cands.append(dict(y=y, a=a, b=b, drops=drops))  # ≥2 分叉才是总线
    merged = []                                                             # 合并相邻行的同一条总线
    for m in sorted(cands, key=lambda d: -len(d["drops"])):                 # 分叉多的先登记
        for o in merged:                                                    # 与已登记的比较
            if abs(o["y"] - m["y"]) <= 6 and not (m["b"] < o["a"] - 25 or m["a"] > o["b"] + 25):  # 纵向相邻且横向重叠
                o["a"], o["b"] = min(o["a"], m["a"]), max(o["b"], m["b"])   # 扩展范围
                o["drops"] = sorted(set(o["drops"] + m["drops"]))           # 合并分叉
                break                                                       # 归并完成
        else: merged.append(dict(m))                                        # 新总线
    out, seen = [], set()                                                   # 父子关系与去重集合
    for m in merged:                                                        # 每条总线
        cx = (m["a"] + m["b"]) / 2                                          # 总线中心 x
        # 子节点用"与总线横向有交叠"判定，而不是"中心落在总线内"：
        # 最外侧节点的中心常在总线端点之外（实测珠海销售处中心 486 落在总线起点 502 左侧 16px），
        # 用中心判定会把它漏掉，答案就会从 6 个变成 5 个。
        kids = [l for l in lines if m["y"] < l[2] < m["y"] + 220             # 总线正下方一格内
                and l[1] <= m["b"] + 15 and l[3] >= m["a"] - 15]             # 横向与总线有交叠
        # 父节点用"纵向最近"判定：同一横坐标附近常有多层框（股东大会/总经理），
        # 真正相连的是紧挨总线上方的那一个，取 y2 最大者最稳。
        near = [l for l in lines if l[4] < m["y"] - 5 and abs((l[1] + l[3]) / 2 - cx) < W * 0.12]  # 上方且横向邻近
        if not near or len(kids) < 2: continue                              # 缺父或不足 2 个子节点则丢弃
        par = max(near, key=lambda l: l[4])                                 # 取最靠近总线的父块
        names = [k[0] for k in sorted(kids, key=lambda k: k[1])]            # 按横坐标排序，保持图面顺序
        key = (par[0], tuple(names))                                        # 去重键
        if key not in seen: seen.add(key); out.append((par[0], names))       # 收下未出现过的关系
    return out                                                              # 返回父子关系


def extract(path, doc: str = "", page: int = 1, caption: str = "") -> dict:   # ★单张图 → 可检索语义块
    """图像解析总入口：CLIP 定类型 + OCR 读文字 + 几何还原层级，合成一段自带出处的检索文本。

    返回块的 text 形如：
      【图像语义】类型：组织结构图（CLIP 0.999）
      【图内文字】销售部 | 渠道销售部 | 大客户销售部 | 珠海销售处 | …
      【图内层级】销售部 下辖：电话及网络销售部、渠道销售部、大客户销售部、国际贸易部
    """
    try:                                                                    # 单图失败不能拖垮整篇解析
        img = Image.open(path).convert("RGB")                               # 读图
        if not box_ok(img.size): return {}                                  # 过小的图直接丢弃
        itype, conf = classify(img)                                         # ★CLIP 语义类型
        lines = read_lines(img)                                             # ★OCR 图内文字
        if not lines and itype in ("产品实物照片", "人物照片", "公司标识logo", "手写签名", "公司印章"):  # 无文字的装饰类图
            pass                                                            # 仍然保留，语义类型本身即信息
        gray = np.array(img.convert("L"))                                   # 灰度图
        dark = gray < 140                                                   # 二值化：招股书的框图线条为深色
        rel = relations(dark, lines) if itype in TREE_TYPES else []          # ★几何层级还原，仅对结构框图启用
        words = [l[0] for l in sorted(lines, key=lambda l: (l[2], l[1]))]    # 图内文字按阅读顺序（先上后左）
        parts = [f"【图像语义】类型：{itype}（CLIP 置信度 {conf}）"]             # 第一行：多模态语义判定
        if words: parts.append("【图内文字】" + " | ".join(words)[:C.IMAGE_OCR_MAX])  # 第二行：图内全部文字
        pr = num_pairs(lines, img.width)                                     # ★图表标签-数值配对
        if pr: parts.append("【图内数值】" + "、".join(pr))                      # 显式给出配对，避免数字张冠李戴
        for par, kids in rel[:12]:                                          # 第三部分：逐条层级关系
            parts.append(f"【图内层级】{par} 下辖：{'、'.join(kids)}（共 {len(kids)} 个）")  # 显式给出数量，避免模型数错
        if caption: parts.insert(0, f"【所在位置】《{doc}》第 {page} 页，图前正文：{caption}")  # 语境，助检索定位
        return dict(text="\n".join(parts)[:C.IMAGE_BLOCK_MAX],              # 合成块文本
                    block_type="image", image_type=itype, image_path=str(path),  # 块类型与元数据
                    page_no=page, page_end=page,                            # 页码
                    locator=f"{doc} 第{page}页 图像·{itype}")                # 可读定位串
    except Exception as e:                                                  # 图像损坏、OCR/CLIP 异常
        print(f"[图像解析降级] {path}: {type(e).__name__}: {e}")               # 打印原因
        return {}                                                           # 返回空块，由文本块兜底
