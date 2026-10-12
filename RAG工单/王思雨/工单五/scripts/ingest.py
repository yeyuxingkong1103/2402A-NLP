# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：知识库入库脚本。解析两份招股说明书 PDF → 图像内容补文本 → 语义分块 → 向量入库 + BM25 建索引。
# 用法：python scripts/ingest.py [--no-images] [--limit-images N]
#   --no-images    跳过图像 OCR（快速重建纯文本索引）
#   --limit-images 仅对前 N 张图做 OCR（调试用）
import argparse                                                                 # 命令行参数
import json                                                                     # 落盘
import sys                                                                      # 调整导入路径
import time                                                                     # 计时
from pathlib import Path                                                        # 路径

ROOT = Path(__file__).resolve().parent.parent                                   # 项目根目录
sys.path.insert(0, str(ROOT))                                                   # 让 backend 包可被导入

from backend import config as C                                                 # 复用集中配置
from backend import bm25_store, image_struct, milvus_store, pdf_parser          # 各能力模块
from backend.ocr_image import enrich_scanned_blocks                              # 扫描页兜底
from backend.semantic_chunker import chunk_blocks                                # 语义分块
from backend.text_clean import clean_text, is_noisy                              # 清洗与污染判定

_MIN_IMG_PX = 200                                                               # 小于该边长的图多为水印碎片/装饰线，不做 OCR


def _enrich_images(blocks: list, limit_images: int = 0) -> tuple:               # 为图像块补文本
    """对图像/图表块做 OCR 或层级还原；返回 (处理的块数, 用时秒)。

    结果按"图片名 + 尺寸 + 修改时间"缓存到 data/image_texts.json：
    图像阶段实测约 160 秒，重建索引时不该重复付这个代价（长任务先落盘，中断可复用）。
    """
    cache_file = C.ROOT / "data" / "image_texts.json"                           # 缓存文件
    cache = {}                                                                  # 缓存内容
    try:                                                                        # 读缓存可能失败
        if cache_file.exists():                                                 # 文件存在
            cache = json.loads(cache_file.read_text(encoding="utf-8"))          # 读取
    except Exception as exc:                                                    # 文件损坏
        print(f"  [图像] 缓存读取失败，将全部重算：{type(exc).__name__}: {exc}")     # 明确报因
        cache = {}                                                              # 丢弃坏缓存
    todo = [b for b in blocks if b.get("block_type") in ("image", "chart")       # 只看图像类块
            and b.get("image_path") and not b.get("text")]                       # 且当前没有文本
    if limit_images:                                                            # 调试模式
        todo = todo[:limit_images]                                              # 只处理前 N 张
    t0 = time.time()                                                            # 计时
    done, reused = 0, 0                                                         # 计数：新识别 / 复用缓存
    for i, b in enumerate(todo, 1):                                             # 逐张处理
        p = Path(b["image_path"])                                               # 图片路径
        if not p.exists():                                                      # 文件不存在（MinerU 未导出）
            continue                                                            # 跳过
        try:                                                                    # 读图可能失败
            from PIL import Image                                               # 延迟导入
            with Image.open(p) as im:                                           # 打开看尺寸
                if min(im.size) < _MIN_IMG_PX:                                   # 太小的图跳过
                    continue                                                    # 继续下一张
                st = p.stat()                                                   # 取文件状态
        except Exception:                                                       # 打不开
            continue                                                            # 跳过
        ckey = f"{p.name}:{st.st_size}:{int(st.st_mtime)}"                       # 缓存键
        if ckey in cache:                                                       # 命中缓存
            txt = cache[ckey]                                                   # 复用结果
            reused += 1                                                         # 计数
        else:                                                                   # 未命中
            txt = image_struct.image_to_text(str(p))                            # 结构化还原或整图 OCR
            txt = clean_text(txt) if txt else ""                                # 再洗一遍水印
            cache[ckey] = txt                                                   # 写入缓存
            done += 1                                                           # 计数
        if txt and len(txt) >= 10 and not is_noisy(txt):                        # 有效内容才收录
            b["text"] = txt                                                     # 写回块
        if i % 20 == 0:                                                         # 每 20 张报一次进度
            print(f"  [图像] 已处理 {i}/{len(todo)}，用时 {time.time() - t0:.0f}s")  # 进度
    try:                                                                        # 写缓存可能失败
        cache_file.write_text(json.dumps(cache, ensure_ascii=False),            # 落盘
                              encoding="utf-8")                                 # 指定编码
    except Exception as exc:                                                    # 写失败
        print(f"  [图像] 缓存写入失败：{type(exc).__name__}: {exc}")              # 只告警，不影响入库
    print(f"  [图像] 新识别 {done} 张，复用缓存 {reused} 张")                        # 统计
    return done, time.time() - t0                                               # 返回统计


def _build_entities(blocks: list) -> list:                                      # 抽取公司实体
    """从正文里统计"股份有限公司/有限公司"形态的名字，取出现次数靠前的若干作为实体表。"""
    import re                                                                   # 正则
    from collections import Counter                                             # 计数
    pat = re.compile(r"[一-龥A-Za-z（）()]{2,30}?(?:股份有限公司|有限责任公司|有限公司)")  # 全称形态
    cnt = Counter()                                                             # 计数器
    for b in blocks:                                                            # 遍历块
        if b.get("block_type") in ("header", "page_number"):                    # 页眉页脚不计
            continue                                                            # 跳过
        for m in pat.findall(b.get("text", "") or ""):                          # 逐个人名式匹配
            if len(m) >= 6:                                                     # 太短的多是误匹配
                cnt[m] += 1                                                     # 计数
    names = [n for n, c in cnt.most_common(20) if c >= 2]                       # 出现 2 次以上才算实体
    return names                                                                # 返回


def main() -> int:                                                              # 主流程
    """执行完整入库；返回进程退出码。"""
    ap = argparse.ArgumentParser(description="工单五 知识库入库")                   # 参数解析器
    ap.add_argument("--no-images", action="store_true", help="跳过图像 OCR")       # 跳过图像
    ap.add_argument("--limit-images", type=int, default=0, help="仅处理前 N 张图")  # 限量
    args = ap.parse_args()                                                      # 解析

    t_all = time.time()                                                         # 总计时
    pdfs = C.raw_pdfs()                                                         # 知识库 PDF
    if not pdfs:                                                                # 没有文件
        print(f"[入库] {C.RAW_DIR} 下没有 PDF，请先放入招股说明书")                  # 提示
        return 1                                                                # 失败退出

    all_blocks = []                                                             # 全部块
    for pdf in pdfs:                                                            # 逐篇解析
        print(f"[入库] 解析 {pdf.name} …")                                        # 进度
        t0 = time.time()                                                        # 计时
        blocks = pdf_parser.parse_pdf(pdf)                                      # 解析
        blocks = enrich_scanned_blocks(blocks, pdf)                             # 扫描页补 OCR
        print(f"  完成：{len(blocks)} 块，用时 {time.time() - t0:.0f}s")             # 进度
        all_blocks.extend(blocks)                                               # 合并

    if not args.no_images:                                                      # 未跳过图像
        # 先把 torch/向量模型拉起来，再进 OCR。
        # 坑（实测）：Windows 上先导入 paddleocr 再导入 torch 会报 WinError 127（shm.dll 加载失败），
        # 原因是两者的 MKL/OpenMP DLL 冲突且顺序敏感；实测"先 torch 后 paddle"稳定。
        print("[入库] 预加载向量模型（同时规避 torch/paddle 导入顺序冲突）…")         # 进度
        milvus_store.embed_model()                                              # 预加载 bge-m3
        print("[入库] 图像内容提取（组织结构图/流程图会做层级还原）…")                  # 进度
        done, sec = _enrich_images(all_blocks, args.limit_images)               # 处理
        print(f"  完成：{done} 张图产出有效文本，用时 {sec:.0f}s")                    # 进度

    entities = _build_entities(all_blocks)                                      # 抽实体表
    (C.ROOT / "data").mkdir(parents=True, exist_ok=True)                        # 确保目录存在
    (C.ROOT / "data" / "entities.json").write_text(                             # 落盘实体表
        json.dumps(entities, ensure_ascii=False, indent=1), encoding="utf-8")   # 供 Query 改写使用
    print(f"[入库] 实体表：{len(entities)} 个公司名")                              # 进度

    print("[入库] 语义分块（chunk_size=%d, overlap=%d）…" % (C.CHUNK_SIZE, C.CHUNK_OVERLAP))  # 进度
    chunks = chunk_blocks(all_blocks, milvus_store.embed)                       # 分块（编码用于语义切分）
    print(f"  完成：{len(chunks)} 个分块")                                        # 进度

    C.INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)                      # 确保目录存在
    C.INDEX_FILE.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")  # 落盘分块
    print(f"[入库] 分块已落盘：{C.INDEX_FILE}")                                    # 进度

    print("[入库] 写入 Milvus …")                                                 # 进度
    try:                                                                        # Milvus 可能不可用
        client = milvus_store.milvus_client()                                   # 建客户端
        milvus_store.recreate_collection(client)                                # 重建集合，保证干净
        n = milvus_store.insert_chunks(client, chunks)                          # 写入
        print(f"  完成：写入 {n} 条向量")                                           # 进度
    except Exception as exc:                                                    # 写入失败
        print(f"  [警告] Milvus 写入失败：{type(exc).__name__}: {exc}")           # 明确报因
        print("         服务仍可用，但向量通道会降级为仅 BM25 关键词检索")            # 说明影响

    print("[入库] 建立 BM25 索引 …")                                              # 进度
    index, _ = bm25_store.build(chunks)                                         # 建索引
    bm25_store.save(index, chunks)                                              # 落盘
    print(f"  完成：{C.BM25_FILE}")                                              # 进度

    print(f"[入库] 全部完成，总用时 {time.time() - t_all:.0f}s")                     # 总进度
    return 0                                                                    # 成功退出


if __name__ == "__main__":                                                      # 直接执行
    raise SystemExit(main())                                                    # 以返回码退出
