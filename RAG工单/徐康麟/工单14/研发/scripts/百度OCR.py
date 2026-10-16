# -*- coding: utf-8 -*-
"""百度 OCR API 客户端（工单14/15/18 的扫描件解析用）。

工单编号：人工智能NLP-RAG-修复低质量工业PDF 的解析与信息丢失工单（跨工单复用）

为什么用 API：本机无任何本地 OCR 引擎（cv2/pytesseract/paddleocr/easyocr/onnxruntime 全缺），
而百度 OCR 的网络链路已实测可达（`https://aip.baidubce.com/oauth/2.0/token` 返回 401 = 凭据无效，
说明域名与 TLS 均正常）。

## 凭据从哪来（申请步骤）
1. 打开 https://console.bce.baidu.com/ai-engine/ocr/overview/index
   （需百度账号登录；首次使用按提示完成实名认证）
2. 左侧「文字识别」→「概览」页点 **「创建应用」**
3. 应用名称随意（如 `rag-ocr`），接口选 **「通用文字识别（含位置版）」**
   —— 工单14 需要坐标信息来回答"部件相对位置"，故必须选**含位置**的接口
4. 创建后在应用列表复制 **AppID / API Key / Secret Key**

## 凭据怎么给（三选一，本脚本都支持）
* 环境变量：`BAIDU_OCR_API_KEY` / `BAIDU_OCR_SECRET_KEY`
* 命令行：`--api-key ... --secret-key ...`
* 配置文件：`研发/data/百度OCR凭据.json`（形如 `{"api_key": "...", "secret_key": "..."}`，**勿提交**）

## 用法
```powershell
$py = 'E:\gao6gongdan\工单1\.venv\Scripts\python.exe'
# 1) 先自检（不需要密钥，验证网络与脚本）
& $py 研发/scripts/百度OCR.py --selftest
# 2) 单张图识别
& $py 研发/scripts/百度OCR.py --image 研发/data/_render/page07.png --out 研发/data/ocr/page07.json
# 3) 批量识别整页目录
& $py 研发/scripts/百度OCR.py --dir 研发/data/_render --out-dir 研发/data/ocr
```

依赖：仅标准库（urllib + json + base64），无需新装任何包。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request

TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
# 默认用**高精度含位置版**：返回每个文本框坐标（工单14/15 的"部件相对位置"题必需），
# 且实测识别更细（同一附图页 21 行 vs 标准含位置版 15 行）。
# 实测：账号免费额度覆盖 general_basic / accurate / accurate_basic，
#       但**不含** general（通用含位置版）—— 用它会报 error_code=18。
OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate"
# 可选：通用文字识别（含位置版）—— 多数账号需单独领额度
OCR_URL_GENERAL = "https://aip.baidubce.com/rest/2.0/ocr/v1/general"
# 可选：高精度标准版（无坐标，仅文本）
OCR_URL_ACCURATE_BASIC = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"

CRED_FILE = pathlib.Path(r"E:\gao6gongdan\工单14\研发\data\百度OCR凭据.json")
TOKEN_CACHE = pathlib.Path(r"E:\gao6gongdan\工单14\研发\data\_百度OCR_token.json")


def load_credentials(args) -> tuple[str, str]:
    """按 命令行 > 配置文件 > 环境变量 的顺序取凭据。"""
    if args.api_key and args.secret_key:
        return args.api_key, args.secret_key
    if CRED_FILE.exists():
        try:
            d = json.loads(CRED_FILE.read_text(encoding="utf-8"))
            if d.get("api_key") and d.get("secret_key"):
                return d["api_key"], d["secret_key"]
        except Exception:
            pass
    return os.environ.get("BAIDU_OCR_API_KEY", ""), os.environ.get("BAIDU_OCR_SECRET_KEY", "")


def get_token(api_key: str, secret_key: str, *, force: bool = False) -> str:
    """取 access_token（缓存 29 天，百度 token 有效期 30 天）。"""
    if not force and TOKEN_CACHE.exists():
        try:
            c = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
            if c.get("token") and c.get("expire_at", 0) > time.time():
                return c["token"]
        except Exception:
            pass
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key,
    }).encode("utf-8")
    req = urllib.request.Request(TOKEN_URL, data=body,
                                headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode("utf-8"))
    if "access_token" not in d:
        raise RuntimeError(f"取 token 失败：{d}")
    TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_CACHE.write_text(json.dumps(
        {"token": d["access_token"], "expire_at": time.time() + 29 * 86400}), encoding="utf-8")
    return d["access_token"]


def ocr_image(token: str, image_path: pathlib.Path, *, accurate: bool = True,
              timeout: int = 120, retries: int = 4, backoff: float = 3.0) -> dict:
    """识别单张图片，返回百度原始响应（含 words_result 与坐标）。

    对 QPS/配额类错误（error_code 18 / 17 / 4）自动退避重试 ——
    新创建的应用常有配额尚未生效的短暂窗口。
    """
    raw = image_path.read_bytes()
    # 百度要求 base64 后 < 4MB（约 3MB 原图）
    if len(raw) > 3 * 1024 * 1024:
        return {"error": f"图片过大 {len(raw)} B（>3MB），请先降 DPI 或分块"}
    data = urllib.parse.urlencode({
        "image": base64.b64encode(raw).decode("ascii"),
        "recognize_granularity": "small",   # 小粒度：单字级坐标
        "vertexes_location": "true",        # 返回四角坐标
        "probability": "true",
    }).encode("utf-8")
    url = (OCR_URL if accurate else OCR_URL_GENERAL) + "?access_token=" + token

    last: dict = {}
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                last = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = {"error_code": e.code, "error_msg": f"HTTP {e.code}"}
        except Exception as e:
            last = {"error_code": -1, "error_msg": f"{type(e).__name__}: {e}"}

        code = last.get("error_code")
        if code is None and last.get("words_result") is not None:
            return last
        # 17=日配额用尽 / 18=QPS 超限 / 4=并发超限 —— 退避重试；其余直接返回
        if code in (4, 17, 18) and attempt < retries:
            wait = backoff * attempt
            print(f"    [重试 {attempt}/{retries - 1}] error_code={code} "
                  f"{last.get('error_msg')} → 等 {wait:.0f}s")
            time.sleep(wait)
            continue
        return last
    return last


def extract_text(resp: dict) -> str:
    """从百度响应里取出纯文本（按返回顺序拼接）。"""
    return "\n".join(str(w.get("words", "")) for w in (resp.get("words_result") or []))


def selftest() -> int:
    """不需要凭据的自检：验网络 + 打印申请步骤。"""
    print("===== 百度 OCR 客户端自检 =====")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  token 端点  : {TOKEN_URL}")
    print(f"  OCR  端点   : {OCR_URL}（含位置版）")
    try:
        body = urllib.parse.urlencode({"grant_type": "client_credentials",
                                       "client_id": "probe", "client_secret": "probe"}).encode()
        req = urllib.request.Request(TOKEN_URL, data=body,
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.loads(r.read().decode("utf-8"))
            print(f"  [网络] 可到百度，响应：{str(d)[:120]}")
        except urllib.error.HTTPError as e:
            code = e.code
            print(f"  [网络] 可到百度 ✅（HTTP {code}"
                  f"{' = 凭据无效，说明链路正常' if code in (400, 401) else ''}）")
        except Exception as e:
            print(f"  [网络] 失败：{type(e).__name__}: {e}")
            return 3
    except Exception as e:
        print(f"  [网络] 异常：{e}")
        return 3

    ak, sk = load_credentials(argparse.Namespace(api_key=None, secret_key=None))
    print(f"  [凭据] " + ("已提供（来源：配置文件或环境变量）" if ak and sk else
                          "**未提供** —— 请按下方步骤申请后填入"))
    print("\n===== 申请步骤 =====")
    print("  1. 打开 https://console.bce.baidu.com/ai-engine/ocr/overview/index（百度账号登录）")
    print("  2. 左侧「文字识别」→ 概览页点「创建应用」")
    print("  3. 接口勾选「通用文字识别（含位置版）」—— 位置信息是本工单答题必需")
    print("  4. 复制 AppID / API Key / Secret Key")
    print("  5. 三种提供方式任选：")
    print(f"       a) 写入 {CRED_FILE}")
    print('          {"api_key":"...","secret_key":"..."}')
    print("       b) 环境变量 BAIDU_OCR_API_KEY / BAIDU_OCR_SECRET_KEY")
    print("       c) 命令行 --api-key ... --secret-key ...")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="百度 OCR API 客户端（扫描件解析）")
    ap.add_argument("--selftest", action="store_true", help="自检网络与凭据（不需密钥）")
    ap.add_argument("--image", help="单张图片路径")
    ap.add_argument("--dir", help="批量：整目录 PNG")
    ap.add_argument("--out", help="单张结果 JSON 路径")
    ap.add_argument("--out-dir", help="批量结果目录")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--secret-key", default=None)
    ap.add_argument("--general", action="store_true", help="改用通用含位置版（需单独领额度，多数账号未开通）")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    ak, sk = load_credentials(args)
    if not ak or not sk:
        print("[!!] 未提供百度 OCR 凭据。运行 --selftest 查看申请步骤。")
        return 2
    try:
        token = get_token(ak, sk)
    except Exception as e:
        print(f"[!!] 取 token 失败：{e}")
        return 3
    print(f"[OK] token 就绪（缓存于 {TOKEN_CACHE.name}）")

    targets: list[pathlib.Path] = []
    if args.image:
        targets.append(pathlib.Path(args.image))
    if args.dir:
        targets.extend(sorted(pathlib.Path(args.dir).glob("*.png")))
    if not targets:
        print("[!!] 请给 --image 或 --dir")
        return 2

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    for fp in targets:
        if not fp.exists():
            print(f"  [跳过] {fp} 不存在")
            continue
        try:
            resp = ocr_image(token, fp, accurate=(not args.general))
        except Exception as e:
            print(f"  [失败] {fp.name}: {type(e).__name__}: {e}")
            fail += 1
            continue
        if "error_code" in resp:
            print(f"  [失败] {fp.name}: error_code={resp.get('error_code')} {resp.get('error_msg')}")
            fail += 1
            continue
        n = len(resp.get("words_result") or [])
        txt = extract_text(resp)
        dst = pathlib.Path(args.out) if args.out else (out_dir / f"{fp.stem}.ocr.json" if out_dir else None)
        if dst:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps({
                "image": str(fp), "识别行数": n, "字符数": len(txt),
                "text": txt, "raw": resp,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] {fp.name}: {n} 行 / {len(txt)} 字符" + (f" -> {dst}" if dst else ""))
        ok += 1

    print(f"\n完成：成功 {ok} / 失败 {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
