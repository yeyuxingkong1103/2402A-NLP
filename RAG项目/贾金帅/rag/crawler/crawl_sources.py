# -*- coding: utf-8 -*-
"""各数据源的调度逻辑：把 spiders 抓到的结果落到 raw/。

自 ``pipeline.py`` 拆出。每个源一个 ``run_*`` 函数，统一登记在 ``RUNNERS``：
新增数据源 = 在 spiders 写抓取逻辑 -> 这里加一个 run_* -> 登记一行。
"""
import argparse
import json
import os
import time

from .base import get_logger, http_get, now_iso
from .crawl_storage import (_count_docs, _safe_filename, otc_done,
                            record_nmpa_failure, sink_doc)
from .schema import (PATHS, build_doc, done_ids, load_lines, prune_failures,
                     record_failure, safe_id, save_json)
from .spiders import (ChiCTRSpider, NMPASpider, OtcAnnSpider, WanfangSpider,
                      YibaoCatalog, ann_id_from_url, build_output, extract_pdf,
                      resolve_term_cn)


# 连续多少个药名抓不到批准文号就熔断（整体失效时别继续空跑）
EMPTY_STREAK_LIMIT = 20


# 每爬 N 个药名主动校验一次会话；瑞数令牌有 TTL，跑久了会静默失效
SESSION_REFRESH_EVERY = 5


def extract_seed_names(path):
    """从种子文件抽去重药名清单（默认吃 drugs_standard.json 的 drug_name 字段）。

    支持 list[dict] 与 {"drugs":[dict]} 两种形态；保序去重。
    返回 [] 表示没有可用种子，上层应提示无法一把梭。
    """
    if not path or not os.path.exists(path):
        return []
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        get_logger().warning("种子文件读取失败：%s", e)
        return []
    rows = data if isinstance(data, list) else []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                rows = v
                break
    seen, out = set(), []
    for it in rows:
        if isinstance(it, dict) and it.get("drug_name"):
            n = it["drug_name"].strip()
            if n and n not in seen:
                seen.add(n)
                out.append(n)
    return out


# ======================= 各数据源的执行逻辑 =======================
def run_nmpa(args):
    """药品批准文号：逐个药名抓，结果追加进进度 jsonl（天然断点续爬）。"""
    logger = get_logger()
    progress, out = PATHS["nmpa_progress"], PATHS["nmpa_out"]
    yibao_path = getattr(args, "yibao", None)
    done = set()
    for ln in load_lines(progress):          # 逐行解析，坏行跳过而不是整体崩
        try:
            done.add(json.loads(ln)["keyword"])
        except Exception:  # noqa: BLE001
            continue
    todo = [n for n in args.items if n not in done]
    if not todo:
        logger.info("全部 %d 个药名已抓过，跳过（清空 %s 可重抓）", len(args.items), progress)
    else:
        # 医保目录：本地有就解析，没有就尝试下载；全失败会明确告警（不再是静默留空）
        YibaoCatalog.ensure(yibao_path)
        # 浏览器初始化 + 过瑞数可能受 IP 限流影响（短时空白页），整体重试几次并拉长冷却
        spider = None
        for outer in range(3):
            spider = NMPASpider(interval=args.interval)
            try:
                spider.start()
                spider.load_home()      # 内部已带递增冷却重试
                spider.check_health()   # itemId 失效时直接抛错，别白跑几小时
                break
            except Exception as e:  # noqa: BLE001
                logger.warning("初始化/过瑞数失败（%s），180s 后整体重试（%d/3）", e, outer + 1)
                try:
                    spider.close()
                except Exception:  # noqa: BLE001
                    pass
                spider = None
                if outer < 2:
                    time.sleep(180)
        if spider is None:
            raise RuntimeError(
                "NMPA 初始化多次失败（疑似瑞数持续限流或 itemId 过期），"
                "请稍后重试，或检查网络/EDGE_PATH。")
        try:
            empty_streak = 0
            since_refresh = 0
            for name in todo:
                try:
                    # 主动保活：每 N 个药名校验一次会话，失效就重建（防瑞数静默过期）
                    if since_refresh >= SESSION_REFRESH_EVERY:
                        if not spider.verify_session():
                            spider._refresh_session()
                        since_refresh = 0
                    rec = spider.crawl_drug(name)
                except Exception as e:  # noqa: BLE001
                    logger.error("[%s] 抓取失败：%s", name, e)
                    continue
                if not rec.get("records"):
                    # 0 记录：先用探针确认会话是否还活着。活=真没有；死=刷新后重试一次
                    if not spider.verify_session():
                        logger.warning("[%s] 会话已失效（探针也查不到），刷新后重试", name)
                        if spider._refresh_session():
                            try:
                                rec = spider.crawl_drug(name)
                            except Exception as e:  # noqa: BLE001
                                logger.error("[%s] 重试抓取失败：%s", name, e)
                    if not rec.get("records"):
                        # 抓不到任何批准文号：记失败清单，不写进度（下次还能重试）
                        empty_streak += 1
                        record_nmpa_failure(name, rec["info"].get("监管分类依据", "无批准文号记录"))
                        logger.warning("[%s] 无批准文号记录，已记入失败清单", name)
                        if empty_streak >= EMPTY_STREAK_LIMIT:
                            raise RuntimeError(
                                "连续 %d 个药名都抓不到批准文号记录，疑似接口失效或 "
                                "itemId 过期，已中止。请核对网络/反爬状态后重跑。"
                                % empty_streak)
                        continue
                empty_streak = 0
                since_refresh += 1
                with open(progress, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        finally:
            spider.close()

    if args.merge:
        merge_nmpa(progress, out, yibao_path)


def merge_nmpa(progress, out, yibao_path=None):
    """把进度 jsonl 合并去重为 drugs.json（按 药品通用名+国药准字 去重）

    抓不到记录的药名不进 drugs.json，改为记进 nmpa_failed.jsonl —— 以前它们
    会变成全空记录混在里面（实测 26 条），下游分不清「抓失败」和「确实没有」。
    """
    logger = get_logger()
    lookup = YibaoCatalog.ensure(yibao_path)
    items, seen, failed = [], set(), []
    for line in load_lines(progress):
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        produced = build_output([rec], lookup)
        if not produced:
            failed.append(rec.get("keyword", ""))
            continue
        for it in produced:
            key = (it.get("药品通用名"), it.get("国药准字"))
            if key in seen:
                continue
            seen.add(key)
            items.append(it)
    save_json(out, items)
    for kw in failed:
        record_nmpa_failure(kw, "无批准文号记录")
    if failed:
        logger.warning("%d 个药名抓不到批准文号，已记入 %s",
                       len(set(failed)), PATHS["nmpa_failed"])
    logger.info("已合并 %d 条记录 -> %s", len(items), out)


def _run_browser_terms(args, spider_cls, source):
    """chictr / wanfang 通用流程：浏览器爬虫按检索词逐条 yield doc，统一落盘。"""
    logger = get_logger()
    out_dir = PATHS[source]
    os.makedirs(out_dir, exist_ok=True)
    done = done_ids(out_dir)
    spider = spider_cls(interval=args.interval)
    spider.start()
    stat = {"ok": 0, "skip": 0, "fail": 0}
    try:
        for term in args.items:
            cn = resolve_term_cn(term) if source == "chictr" else term
            logger.info("[%s] 检索：%s", source, cn)
            try:
                for doc in spider.crawl_term(cn, args.max or 30, done):
                    r = sink_doc(out_dir, doc)
                    stat[r] += 1
                    done.add(safe_id(doc["doc_id"]))
            except Exception as e:  # noqa: BLE001
                logger.error("[%s] %s 中断：%s", source, cn, e)
            logger.info("[%s] 累计 ok=%d skip=%d fail=%d",
                        cn, stat["ok"], stat["skip"], stat["fail"])
    finally:
        spider.close()
    prune_failures(out_dir)


def run_chictr(args):
    _run_browser_terms(args, ChiCTRSpider, "chictr")


def run_wanfang(args):
    _run_browser_terms(args, WanfangSpider, "wanfang")


def run_guideline(args):
    """指南：urls 文件每行 `doc_id<TAB>url[<TAB>标题]`，下载 PDF 抽全文后落盘。"""
    out_dir = PATHS["guideline"]
    pdf_dir = os.path.join(out_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    done = done_ids(out_dir)

    for line in args.items:
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) < 2:
            continue
        doc_id, url = parts[0], parts[1]
        title = parts[2] if len(parts) > 2 else doc_id
        if safe_id(doc_id) in done:
            continue
        get_logger().info("[guideline] 下载：%s", title)
        # 先落本地再抽文本：抽失败也能保留原件，且可离线重抽
        pdf_path = os.path.join(pdf_dir, doc_id.replace("/", "_") + ".pdf")
        if not os.path.exists(pdf_path):
            resp = http_get(url, timeout=60)
            if not resp or not resp.content:
                record_failure(out_dir, doc_id, "下载失败")
                continue
            with open(pdf_path, "wb") as f:
                f.write(resp.content)
        text = extract_pdf(pdf_path)
        if len(text) < 200:
            record_failure(out_dir, doc_id, "正文不足200字")
            continue
        sink_doc(out_dir, build_doc(
            "guideline", doc_id, url, title, "", {"正文": text}, text, [],
            doc_id=doc_id))
    prune_failures(out_dir)


def run_otc_ann(args):
    """处方药转 OTC 公告：浏览器加载列表/公告页拿链接，http 直下官方附件原件。

    产物：data/raw/otc_ann/{ann_id}/ 下每个公告一个子目录，内含
      - 原始附件（.docx/.doc/.pdf，官网给啥存啥，不转格式）
      - index.json（元数据：标题/发布日期/来源URL/附件清单/抓取时间）
    """
    logger = get_logger()
    out_dir = PATHS["otc_ann"]
    os.makedirs(out_dir, exist_ok=True)
    done = otc_done(out_dir)
    spider = OtcAnnSpider(interval=args.interval)
    spider.start()
    stat = {"ann": 0, "att": 0, "skip": 0, "fail": 0}
    try:
        announcements = spider.list_announcements(max_pages=args.pages or 3)
        logger.info("发现 %d 篇「转换为非处方药」公告", len(announcements))
        for title, url in announcements:
            aid = ann_id_from_url(url)
            if aid in done:
                stat["skip"] += 1
                continue
            if args.max and stat["ann"] >= args.max:
                break
            try:
                t, date, atts = spider.ann_page_attachments(url)
            except Exception as e:  # noqa: BLE001
                logger.error("[%s] 读取失败：%s", title, e)
                stat["fail"] += 1
                continue
            adir = os.path.join(out_dir, aid)
            os.makedirs(adir, exist_ok=True)
            rec_atts = []
            for name, aurl in atts:
                fn = _safe_filename(name) or os.path.basename(aurl.split("?")[0])
                path = os.path.join(adir, fn)
                if os.path.exists(path):          # 同公告内附件去重
                    rec_atts.append({"name": name, "file": fn})
                    continue
                resp = http_get(aurl, retries=3, backoff=2, timeout=60)
                if not resp or not resp.content:
                    logger.warning("附件下载失败：%s", name)
                    stat["fail"] += 1
                    continue
                with open(path, "wb") as f:
                    f.write(resp.content)
                rec_atts.append({"name": name, "file": fn})
                stat["att"] += 1
            index = {
                "doc_id": "otc_ann:" + aid,
                "title": t or title,
                "publish_date": date,
                "source_url": url,
                "attachments": rec_atts,
                "crawl_time": now_iso(),
            }
            save_json(os.path.join(adir, "index.json"), index)
            done.add(aid)
            stat["ann"] += 1
            logger.info("[ok] %s -> %d 个附件", t or title, len(rec_atts))
    finally:
        spider.close()
    logger.info("公告 %d / 附件 %d / 跳过 %d / 失败 %d",
                stat["ann"], stat["att"], stat["skip"], stat["fail"])


RUNNERS = {
    "nmpa": run_nmpa, "chictr": run_chictr,
    "wanfang": run_wanfang, "guideline": run_guideline, "otc_ann": run_otc_ann,
}


# ======================= 一把梭：基于本地种子跑全部可自动源 =======================
def run_all(args):
    """一条命令跑完「能自动爬」的源，断点续爬天然安全。

    顺序：nmpa（批准文号，吃种子药名）→ otc_ann（自动翻页）。
    不自动跑的源（需人工输入，跳过并提示）：
      - guideline：需要 PDF 直链清单（--urls-file），无通用「全部」接口
      - chictr / wanfang：需要检索词（疾病/适应症），非药品维度
    """
    names = extract_seed_names(PATHS["seed_names"])
    if not names:
        raise SystemExit("未找到种子药名（%s）。无法一把梭 nmpa，"
                         "请先准备一份含 drug_name 字段的药名清单。" % PATHS["seed_names"])
    logger = get_logger()
    logger.info("=" * 56)
    logger.info("一把梭模式：种子药名 %d 个（%s）", len(names), PATHS["seed_names"])
    logger.info("=" * 56)

    logger.info(">>> [1/3] NMPA 批准文号")
    run_nmpa(argparse.Namespace(items=names, interval=args.interval,
                                merge=args.merge, yibao=args.yibao))

    logger.info(">>> [2/3] 处方药转 OTC 公告（自动翻页）")
    run_otc_ann(argparse.Namespace(interval=args.interval, pages=args.pages, max=args.max))

    logger.info(">>> 跳过（需人工输入，非药品维度）：")
    logger.info("    - guideline：需 --urls-file（PDF 直链清单）")
    logger.info("    - chictr / wanfang：需 --terms-file（疾病/适应症检索词）")
