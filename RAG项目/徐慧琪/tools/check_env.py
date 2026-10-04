"""③a 的环境自检：把"我以为环境是好的"变成可复核的输出。

存在的理由：③a 依赖六个外部东西（两个容器、两个本地模型、一个本地 LLM、一个外部 API 密钥），
任务 6 起多一项限流加盐值（缺了它公开端点全 500），终审 I-5 起再多一项 JWT 签名密钥
（缺了它服务**起不来**）—— 任何一样没到位，报错都会出现在链路深处且面目模糊，比如缺了
sparse_linear.pt 会让检索安静地退化而不是抛错，缺盐则要等第一个公众请求才响亮。
开工前先跑本脚本，比在 answer.py 里 debug 便宜得多。

用法：cd D:/xinzg6/fl && python tools/check_env.py
"""
from __future__ import annotations

import os
import pathlib
import sys

# 本脚本在 tools/ 下而 app 包在 backend/ 下：模块顶部只挂路径，app.* 的导入
# 一律推迟到各检查函数里 —— 这样单测导入本模块不必拉起 torch / pymilvus
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

MILVUS_COLLECTION = "law_chunks"
EXPECTED_CHUNKS = 3388      # ②期实测值，见设计文档 3.1
EXPECTED_ARTICLES = 1260    # ②期实测值
DEEPSEEK_KEY_ENV = "api"    # 密钥只从环境变量读，不写进代码也不入仓库


def _check_cuda() -> tuple[bool, str]:
    """精排要跑在 GPU 上，否则 30 条候选 3.43s，直接顶穿 AC-12 的 2s。"""
    import torch
    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__} 无可用 CUDA"
    return True, f"torch {torch.__version__} / {torch.cuda.get_device_name(0)}"


def _check_milvus() -> tuple[bool, str]:
    from app.db.milvus import entity_count, get_client
    count = entity_count(get_client(), MILVUS_COLLECTION)
    ok = count == EXPECTED_CHUNKS
    return ok, f"{MILVUS_COLLECTION} 实体数 {count}（期望 {EXPECTED_CHUNKS}）"


def _check_mysql() -> tuple[bool, str]:
    from app.db.mysql import connect
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM article")
            count = cur.fetchone()[0]
    ok = count == EXPECTED_ARTICLES
    return ok, f"article 行数 {count}（期望 {EXPECTED_ARTICLES}）"


def _check_ollama() -> tuple[bool, str]:
    """律师侧生成走本地 Ollama；模型没拉到时 ChatOllama 会到调用时才报错。"""
    import httpx
    resp = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5)
    names = [m["name"] for m in resp.json().get("models", [])]
    ok = any(n.startswith("qwen2.5:3b") for n in names)
    return ok, f"已拉取模型 {len(names)} 个，含 qwen2.5:3b: {ok}"


def _check_deepseek_key() -> tuple[bool, str]:
    """只验存在与非空，绝不打印内容——密钥进日志等于泄露。"""
    value = os.environ.get(DEEPSEEK_KEY_ENV, "")
    return bool(value), f"环境变量 {DEEPSEEK_KEY_ENV} {'已设置' if value else '缺失'}"


def _check_reranker() -> tuple[bool, str]:
    """两个模型目录里该在的文件真在没在。真正的打分能力由 tools/ask.py 与集成测验。

    判定**委托 core/config.py 的 validate()**（检查项名字仍叫 reranker：它被测试
    断言、是一份冻结契约，改名等于改契约）。此前这里自己拼「精排目录 +
    model.safetensors」：那是同一个文件名在本仓库的第二份字面量，改了 core 那边
    而漏了这里，自检会报「缺失」而真装配加载正常 —— 方向恰好相反的假警报；
    委托之后连 FL_* 覆盖也一并生效，自检看的目录与真装配加载的是同一个来源。
    覆盖面顺带扩到编码器目录：缺 sparse_linear.pt 时 FlagEmbedding 会**随机初始化**
    sparse 头（检索安静退化、不报错），比精排缺件（当场加载失败）更该被自检点名。
    """
    from app.core.config import ConfigError, load
    try:
        loaded = load()
        loaded.validate()
    except ConfigError as exc:
        # 校验的报错一次列全缺件，原样交出去比任何自撰措辞都具体；
        # FL_MYSQL_PORT 之类装载期错误也走这支（都是「配置不对」的同一类）
        return False, str(exc)
    return True, f"编码器 {loaded.embed_model_path}；精排 {loaded.reranker_model_path}"


def _check_rate_salt() -> tuple[bool, str]:
    """限流加盐值在不在、够不够长（任务 6；缺了它公开端点全 500）。

    与 deepseek_key 同一条纪律：只报「有没有、多长」，**绝不回显取值** —— 盐进
    日志等于把「不存 IP 原文」这条承诺原地作废。判定**委托** config.load_rate_salt()：
    自检看的规则与限流中间件读的是同一个函数（缺失与过短都抛 MissingRateSaltError），
    长度下限不在这里抄第二份，否则两边漂了自检还报 OK。
    """
    from app.core.config import (ENV_RATE_LIMIT_SALT, MIN_RATE_SALT_LEN,
                                 ConfigError, load_rate_salt)
    try:
        salt = load_rate_salt()
    except ConfigError as exc:
        # 报错文案由 loader 一次说全（缺哪个变量、差多少字符），原样交出去
        return False, str(exc)
    return True, (f"环境变量 {ENV_RATE_LIMIT_SALT} 已设置"
                  f"（{len(salt)} 字符 ≥ {MIN_RATE_SALT_LEN}）")


def _check_jwt_secret() -> tuple[bool, str]:
    """JWT 签名密钥在不在、够不够长（终审 I-5；缺了它服务起不来、登录全废）。

    与 deepseek_key / rate_salt 同一条纪律：只报「有没有、多长」，**绝不回显
    取值** —— 密钥进日志等于泄露。判定**委托** security.load_secret()：lifespan
    启动自检与这里读的是同一个函数，长度下限不抄第二份，否则两边漂了自检还报 OK。
    """
    from app.core.security import (JWT_SECRET_ENV, MIN_SECRET_LEN,
                                   MissingSecretError, load_secret)
    try:
        secret = load_secret()
    except MissingSecretError as exc:
        # 报错文案由 loader 一次说全（缺哪个变量、差多少字符），原样交出去
        return False, str(exc)
    return True, (f"环境变量 {JWT_SECRET_ENV} 已设置"
                  f"（{len(secret)} 字符 ≥ {MIN_SECRET_LEN}）")


# (检查项名, 检查函数)。名字会被测试断言，改名等于改契约
CHECKS = [
    ("cuda", _check_cuda),
    ("milvus", _check_milvus),
    ("mysql", _check_mysql),
    ("ollama", _check_ollama),
    ("deepseek_key", _check_deepseek_key),
    ("reranker", _check_reranker),
    ("rate_salt", _check_rate_salt),
    ("jwt_secret", _check_jwt_secret),
]


def check_all() -> list[tuple[str, bool, str]]:
    """逐个跑检查，异常一律转成"未通过 + 原因"，不中断整轮。"""
    results = []
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as exc:  # 连不上就是未通过，没必要让一个检查炸掉整轮
            ok, detail = False, f"异常：{type(exc).__name__}: {exc}"
        results.append((name, ok, detail))
    return results


def main() -> int:
    results = check_all()
    for name, ok, detail in results:
        print(f"[{'OK ' if ok else 'FAIL'}] {name:14s} {detail}")
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} 通过" + (f"，未通过：{failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
