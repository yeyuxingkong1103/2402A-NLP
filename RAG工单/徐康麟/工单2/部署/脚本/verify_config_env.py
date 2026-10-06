# -*- coding: utf-8 -*-
"""校验 部署/配置/config.example.env 与 研发/app/core/config.py 的字段是否一致。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：部署 / 配置校验

做法：把示例 env 里所有 `RAG_*` 变量注入环境 → 重新加载 Settings →
      检查 config.rejected_env_keys() 是否为空（非空即存在未知字段/类型不匹配），
      并抽查若干关键项是否真的被覆盖生效（证明示例文件不是摆设）。

复现（工作目录 = E:\\gao6gongdan\\工单2）：
    pwsh -NoProfile -File run_py.ps1 -B 部署/脚本/verify_config_env.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / "部署" / "配置" / "config.example.env"
sys.path.insert(0, str(REPO_ROOT / "研发"))

# 只取 RAG_* 行；被注释的（# 开头）跳过。
# 取值处理与 shell/dotenv 语义一致：①去掉成对的首尾引号；②未加引号时去掉行内注释。
def parse_value(raw_value: str) -> tuple[str, bool]:
    """返回 (取值, 是否加过引号)。按 shell/dotenv 语义处理引号与行内注释。"""
    v = raw_value.strip()
    if v[:1] in ('"', "'"):
        quote = v[0]
        end = v.find(quote, 1)
        if end != -1:
            return v[1:end], True  # 引号后的内容（如行内注释）忽略
    if "#" in v:
        v = v.split("#", 1)[0]
    return v.strip(), False


loaded: dict[str, str] = {}
unquoted_with_space: list[str] = []
for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, _, value = line.partition("=")
    key = key.strip()
    if not key.startswith("RAG_"):
        continue
    parsed, quoted = parse_value(value)
    if not quoted and (" " in parsed or "\t" in parsed):
        # bash `source` 会把这种写法当成「赋值 + 命令」，必须加引号
        unquoted_with_space.append(key)
    loaded[key] = parsed

print(f"示例文件: {ENV_FILE}")
print(f"解析到 RAG_* 变量: {len(loaded)} 个")
if unquoted_with_space:
    print("  ❌ 以下取值含空格但未加引号（bash source 会误解析为命令）：")
    for key in unquoted_with_space:
        print(f"     - {key}")

os.environ.update(loaded)

from app.core.config import Settings, rejected_env_keys  # noqa: E402

settings = Settings()  # 重新构造，触发 _apply_env_overrides
rejected = rejected_env_keys(settings)

print("\n--- 未知/类型不匹配的配置项 ---")
if rejected:
    for item in rejected:
        print(f"  ❌ {item}")
else:
    print("  ✅ 无（示例文件全部字段与 config.py 一致）")

print("\n--- 抽查关键项是否生效 ---")
checks = {
    "RAG_RETRIEVAL__TOTAL_BOOST_CAP": (settings.retrieval.total_boost_cap, 1.6),
    "RAG_RETRIEVAL__MIN_CONFIDENCE_COSINE": (settings.retrieval.min_confidence_cosine, 0.55),
    "RAG_RETRIEVAL__RERANK_TOP_N": (settings.retrieval.rerank_top_n, 5),
    "RAG_RERANKER__MODE": (settings.reranker.mode, "auto"),
    "RAG_LLM__BASE_URL": (settings.llm.base_url, "http://127.0.0.1:8000/v1"),
    "RAG_APP__FIRST_TOKEN_BUDGET_SECONDS": (settings.app.first_token_budget_seconds, 3.0),
    "RAG_EMBEDDING__DIMENSION": (settings.embedding.dimension, 1024),
    "RAG_APP__LOG_JSON": (settings.app.log_json, True),
    "RAG_RETRIEVAL__SEGMENT_RECALL_ENABLED": (settings.retrieval.segment_recall_enabled, True),
    "RAG_LLM__ALLOW_EXTRACTIVE_FALLBACK": (settings.llm.allow_extractive_fallback, True),
}
ok = True
for key, (actual, expected) in checks.items():
    good = actual == expected
    ok = ok and good
    print(f"  {'✅' if good else '❌'} {key} = {actual!r}（期望 {expected!r}）")

print("\n--- 一致性结论 ---")
print(f"  变量数: {len(loaded)}；被拒项: {len(rejected)}；未加引号含空格: {len(unquoted_with_space)}；抽查通过: {ok}")
verdict = (not rejected) and ok and (not unquoted_with_space)
print(f"  总体: {'✅ 通过' if verdict else '❌ 不通过'}")
raise SystemExit(0 if verdict else 1)
