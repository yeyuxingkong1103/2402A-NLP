# 工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失
"""把 4 个模型注册进 RAGFlow —— 环境准备脚本，换台机器按顺序跑一遍就行

| 用途 | 模型 | 工厂 | 指向 |
|---|---|---|---|
| 对话 | deepseek-flash | DeepSeek | api.deepseek.com |
| 嵌入 | bge-m3 | OpenAI-API-Compatible | 本机 embedding_server.py 的 /v1 |
| 图片转文字 | deepseek-flash | OpenAI-API-Compatible | api.deepseek.com |
| 重排 | bge-reranker-v2-m3 | OpenAI-API-Compatible | 本机 embedding_server.py 的 /rerank |

两个容易踩的点：
1. add_llm 的字段名是 **api_base**，写成 base_url 会被静默忽略，然后报
   "url cannot be None"。setup_models 老版本就是这么踩的。
2. **deepseek-flash 要走 OpenAI-API-Compatible 再注册一遍 image2text**：
   rag/llm/cv_model.py 里没有 DeepSeek 这个视觉类，直接用 DeepSeek 工厂注册
   会 KeyError。deepseek-flash 本身支持图片输入，API 又是 OpenAI 兼容格式，
   所以借 OpenAI-API-Compatible（对应 OpenAI_APICV）这条路。
   注册完还要把租户默认 img2txt_id 指过去 —— figure_parser 取的是租户默认值。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")   # SystemExit 的报错走 stderr
from ragflow_client import RAGFlow

ENV = Path(r"D:\Pycharm\yzq\zg6\工单14\研发\ragflow\docker\.env")
EMBED_URL = "http://host.docker.internal:9997/v1"    # 容器内访问宿主
RERANK_URL = "http://host.docker.internal:9997"      # RAGFlow 自己会拼 /rerank
VISION_ID = "deepseek-flash___OpenAI-API@OpenAI-API-Compatible"


def read_env(key, default=""):
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    return default


def add(c, label, **body):
    code, j = c._req("POST", "/v1/llm/add_llm", headers=c._h(session=True), json=body)
    ok = j.get("code") == 0
    print(f"  [{'OK ' if ok else '失败'}] {label}"
          f"{'' if ok else ' -> ' + json.dumps(j, ensure_ascii=False)[:200]}")
    return ok


def main():
    key = read_env("DEEPSEEK_API_KEY")
    base = read_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not key:
        raise SystemExit("docker/.env 里没有 DEEPSEEK_API_KEY")

    c = RAGFlow()
    c.load_token()
    c.login()          # /v1/llm、/v1/user 只认登录 token

    code, j = c._req("GET", "/v1/user/info", headers=c._h(session=True))
    tid = j["data"].get("tenant_id") or j["data"]["id"]
    print(f"租户 {tid}\n")

    print("=== 注册模型 ===")
    add(c, "对话 deepseek-flash", llm_factory="DeepSeek", api_key=key,
        llm_name="deepseek-flash", api_base=base, model_type="chat", max_tokens=8192)
    add(c, "嵌入 bge-m3", llm_factory="OpenAI-API-Compatible", api_key="sk-local",
        llm_name="bge-m3", api_base=EMBED_URL, model_type="embedding", max_tokens=8192)
    add(c, "图片转文字 deepseek-flash", llm_factory="OpenAI-API-Compatible", api_key=key,
        llm_name="deepseek-flash", api_base=base, model_type="image2text", max_tokens=8192)
    add(c, "重排 bge-reranker-v2-m3", llm_factory="OpenAI-API-Compatible", api_key="sk-local",
        llm_name="bge-reranker-v2-m3", api_base=RERANK_URL, model_type="rerank", max_tokens=8192)

    print("\n=== 租户默认模型 ===")
    code, j = c._req("POST", "/v1/user/set_tenant_info", headers=c._h(session=True), json={
        "tenant_id": tid, "llm_id": "deepseek-flash@DeepSeek", "embd_id": "bge-m3",
        "asr_id": "", "img2txt_id": VISION_ID,
    })
    print(f"  {'OK ' if j.get('code') == 0 else '失败'} {json.dumps(j, ensure_ascii=False)[:160]}")

    print("\n=== 确认 ===")
    code, j = c._req("GET", "/v1/llm/my_llms", headers=c._h(session=True))
    for factory, v in (j.get("data") or {}).items():
        for m in v.get("llm", []):
            print(f"  {factory:<24} {m['name']:<28} {m['type']}")


if __name__ == "__main__":
    main()
