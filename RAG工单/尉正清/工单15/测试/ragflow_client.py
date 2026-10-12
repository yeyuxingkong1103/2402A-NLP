# 工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失
"""RAGFlow HTTP 客户端 —— 其余测试脚本都建在它上面

为什么要写成脚本而不是点界面：工单要求「多轮测试结果、原因分析及优化方案」，
需要**每一轮都能重跑、结果可比**。点界面没法保证两轮用的是同一套配置。

认证有两套 token，容易混：
    · **SDK API Key**   —— `/api/v1/datasets`、`/api/v1/chats` 这类用，发 `Bearer <key>`
    · **登录 token**    —— `/v1/llm`、`/v1/user` 这类用户态接口用，是 itsdangerous
      签名串，**裸发**，加 `Bearer` 会被当成 API Key 去查表然后 401

用法：
    D:/Anaconda/envs/rag_gd/python.exe ragflow_client.py --action info
    D:/Anaconda/envs/rag_gd/python.exe ragflow_client.py --action bootstrap   # 从零建库
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")   # SystemExit 的报错走 stderr

HERE = Path(__file__).parent
BASE = "http://localhost"
PUB_PEM = HERE / "public.pem"
TOKEN_FILE = HERE / ".api_token"
IDS_FILE = HERE / ".ragflow_ids.json"

EMAIL = "rag15@test.com"    # 工单15 独立租户，不碰工单14 已交付的数据
PASSWORD = "rag15pass123"
NICK = "workorder15"
DS_NAME = "wo15test"
CHAT_NAME = "wo15-chat"
PDF = r"D:\BW\RAG 工单\14-17附件\original_problems\documents\CN100347506C.pdf"


def encrypt_pwd(pwd):
    """RAGFlow 的密码要先 RSA 加密再提交。

    源码在 api/utils/crypt.py:25 —— 前端用 JSEncrypt 做
    `encrypt(Base64.encode(password))`，服务端用 conf/private.pem 解。
    这里用 cryptography 复刻同一步（PKCS1_v1_5 + 先 base64 再加密）。
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    key = serialization.load_pem_public_key(PUB_PEM.read_bytes())
    inner = base64.b64encode(pwd.encode("utf-8"))
    return base64.b64encode(key.encrypt(inner, padding.PKCS1v15())).decode()


class RAGFlow:
    def __init__(self, base=BASE):
        self.base = base.rstrip("/")
        self.s = requests.Session()
        self.api_key = None         # SDK API Key
        self.session_token = None   # 登录 token

    # ---------- 底层 ----------
    def _h(self, auth=True, session=False):
        """session=True 用登录 token（裸发），否则用 SDK API Key（带 Bearer）。"""
        h = {"Content-Type": "application/json"}
        if auth:
            tok = self.session_token if session else self.api_key
            if tok:
                h["Authorization"] = tok if session else f"Bearer {tok}"
        return h

    def _req(self, method, path, **kw):
        kw.setdefault("timeout", 300)
        r = self.s.request(method, f"{self.base}{path}", **kw)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"_raw": r.text[:300]}

    # ---------- 认证 ----------
    def load_token(self):
        """读 测试/.api_token 里的 SDK API Key（直接写进 api_token 表的）。"""
        if TOKEN_FILE.exists():
            self.api_key = TOKEN_FILE.read_text(encoding="utf-8").strip()
            print("  [APIKey] 从文件读取")
            return self.api_key
        return None

    def register(self):
        """注册用户。已存在会失败，忽略即可（接着登录）。"""
        code, j = self._req("POST", "/v1/user/register", headers=self._h(False),
                            json={"email": EMAIL, "password": encrypt_pwd(PASSWORD),
                                  "nickname": NICK})
        print(f"  [注册] HTTP {code} {j.get('message', j.get('code', ''))}")

    def login(self):
        """登录拿用户态 token。

        token 不在响应体里 —— 体里那个 `access_token` 是**登录前**的旧值
        （user_app.py:128 先 to_json() 再 get_uuid()），真正能用的是响应头
        `Authorization`：construct_response(auth=user.get_id()) 塞进去的签名串。
        """
        r = self.s.post(f"{self.base}/v1/user/login", headers=self._h(False),
                        json={"email": EMAIL, "password": encrypt_pwd(PASSWORD)},
                        timeout=60)
        j = r.json()
        if r.status_code != 200 or j.get("code") != 0:
            raise SystemExit(f"登录失败：HTTP {r.status_code} {j}")
        self.session_token = r.headers.get("Authorization")
        if not self.session_token:
            raise SystemExit("登录成功但响应头里没有 Authorization")
        print("  [登录] 成功")
        return self.session_token

    def new_token(self):
        """生成 SDK API Key 并落盘（load_token 之后就能直接读）。"""
        code, j = self._req("POST", "/v1/system/new_token", headers=self._h(session=True))
        if code != 200 or j.get("code") != 0:
            raise SystemExit(f"生成 API Key 失败：HTTP {code} {j}")
        self.api_key = j["data"]["token"]
        TOKEN_FILE.write_text(self.api_key, encoding="utf-8")
        print("  [APIKey] 已生成并写入 .api_token")
        return self.api_key

    # ---------- 知识库 ----------
    def create_dataset(self, name=DS_NAME, chunk_method="naive",
                       # 必须写成 tenant_llm 表里的全名：add_llm 给 OpenAI-API-Compatible
                       # 工厂的模型加了 `___OpenAI-API` 后缀，只写 bge-m3 会报
                       # "Unsupported model"；只写 bge-m3@工厂 会报格式不对
                       embedding_model="bge-m3___OpenAI-API@OpenAI-API-Compatible"):
        # 先按名查。**不能**先 POST —— 重名时 RAGFlow 不报错，而是自动改名成
        # `名字(1)`、`名字(2)` 再建一个，结果重复建库（实测建出过三个同名库）。
        # GET /api/v1/datasets/{id} 在这版上是 405，只能用列表接口带 name 过滤。
        code, j = self._req("GET", "/api/v1/datasets", headers=self._h(),
                            params={"name": name})
        for d in (j.get("data") or []):
            if d.get("name") == name:
                print(f"  [建库] 复用已有：{name}")
                return d["id"]
        body = {"name": name, "chunk_method": chunk_method,
                "embedding_model": embedding_model}
        code, j = self._req("POST", "/api/v1/datasets", headers=self._h(), json=body)
        if code == 200 and j.get("code") == 0:
            print(f"  [建库] {name}（{chunk_method}）")
            return j["data"]["id"]
        raise SystemExit(f"建库失败：HTTP {code} {j}")

    def update_dataset(self, ds_id, **kw):
        code, j = self._req("PUT", f"/api/v1/datasets/{ds_id}", headers=self._h(), json=kw)
        print(f"  [改配置] HTTP {code} {j.get('message','')} {kw}")
        return j

    def upload(self, ds_id, pdf_path=PDF):
        # 这里**不能**用 self._h()：它带 Content-Type: application/json，
        # 会把 requests 的 multipart 设置顶掉，服务端报 "No file part!"。
        # 只带 Authorization 即可，Content-Type 交给 requests 自己生成 boundary。
        h = {k: v for k, v in self._h().items() if k != "Content-Type"}
        with open(pdf_path, "rb") as fh:
            code, j = self._req(
                "POST", f"/api/v1/datasets/{ds_id}/documents",
                headers=h,
                files={"file": (Path(pdf_path).name, fh, "application/pdf")})
        if code == 200 and j.get("code") == 0:
            docs = j["data"]
            doc = docs[0] if isinstance(docs, list) else docs
            print(f"  [上传] {Path(pdf_path).name} -> {doc['id'][:12]}")
            return doc["id"]
        raise SystemExit(f"上传失败：HTTP {code} {j}")

    def parse(self, ds_id, doc_ids):
        code, j = self._req("POST", f"/api/v1/datasets/{ds_id}/chunks",
                            headers=self._h(), json={"document_ids": doc_ids})
        print(f"  [解析] HTTP {code} {j.get('message','')}")
        return j

    def doc_status(self, ds_id, doc_id):
        code, j = self._req("GET", f"/api/v1/datasets/{ds_id}/documents",
                            headers=self._h(), params={"id": doc_id})
        d = j.get("data", {})
        docs = d.get("docs", d) if isinstance(d, dict) else d
        if isinstance(docs, list) and docs:
            return docs[0]
        return d

    def chunks(self, ds_id, doc_id):
        code, j = self._req("GET",
                            f"/api/v1/datasets/{ds_id}/documents/{doc_id}/chunks",
                            headers=self._h(), params={"page_size": 100})
        d = j.get("data", {})
        return d.get("chunks", []) if isinstance(d, dict) else []

    # ---------- 对话 ----------
    def create_chat(self, name=CHAT_NAME, ds_ids=()):
        code, j = self._req("POST", "/api/v1/chats", headers=self._h(),
                            json={"name": name, "dataset_ids": list(ds_ids)})
        if code == 200 and j.get("code") == 0:
            print(f"  [助手] {name}")
            return j["data"]["id"]
        code, j = self._req("GET", "/api/v1/chats", headers=self._h(),
                            params={"name": name})
        if j.get("data"):
            print(f"  [助手] 复用已有：{name}")
            return j["data"][0]["id"]
        raise SystemExit(f"建助手失败：HTTP {code} {j}")

    def ask(self, chat_id, question, timeout=180):
        """走 OpenAI 兼容接口提问，返回 (答案, 耗时秒)。

        question 可以是字符串，也可以是完整的 messages 列表。
        传列表是为了把「检索查询」和「作答要求」分开：RAGFlow 拿最后一轮用户消息
        去做检索，如果把选项和「只回答选项字母」这类通用指令塞在同一条里，
        查询会被这些通用词带偏，召回直接垮掉（实测第 1~4 题从 10 条掉到 1 条）。

        注：这个接口的响应里**没有**引用片段字段（只有 choices/usage），
        要展示召回内容得走 /api/v1/retrieval，见 演示检索.py。
        """
        msgs = question if isinstance(question, list) else [{"role": "user", "content": question}]
        t0 = time.time()
        code, j = self._req(
            "POST", f"/api/v1/chats_openai/{chat_id}/chat/completions",
            headers=self._h(),
            json={"model": "model", "messages": msgs, "stream": False},
            timeout=timeout)
        el = time.time() - t0
        if code != 200:
            return f"[HTTP {code}] {j}", el
        try:
            return j["choices"][0]["message"]["content"], el
        except (KeyError, IndexError):
            return f"[异常响应] {json.dumps(j, ensure_ascii=False)[:200]}", el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="info", choices=["info", "bootstrap"])
    args = ap.parse_args()

    c = RAGFlow()
    if not c.load_token():
        c.register()
        c.login()
        c.new_token()

    if args.action == "bootstrap":
        if IDS_FILE.exists():
            raise SystemExit(
                f"{IDS_FILE.name} 已存在，说明知识库早就建好了；"
                "再跑一次会往里重复上传同一份 PDF。要重建请先删掉该文件。")
        ds = c.create_dataset()
        doc = c.upload(ds)
        c.parse(ds, [doc])
        # 必须等解析出分片再建助手：带 dataset_ids 建助手时会校验
        # "The dataset doesn't own parsed file"（api/apps/sdk/chat.py:161）
        print("  等首次解析完成 ...")
        for _ in range(120):
            st = c.doc_status(ds, doc)
            if st.get("progress", 0) >= 1.0:
                print(f"  解析完成，chunk 数 {st.get('chunk_count')}")
                break
            if st.get("progress", 0) < 0:
                raise SystemExit(f"解析失败：{(st.get('progress_msg') or '')[-500:]}")
            time.sleep(5)
        else:
            raise SystemExit("解析超时")
        chat = c.create_chat(ds_ids=[ds])
        IDS_FILE.write_text(json.dumps(
            {"chat_id": chat, "ds_id": ds, "doc_id": doc}, ensure_ascii=False),
            encoding="utf-8")
        print(f"\n  已写入 {IDS_FILE.name}：ds={ds[:12]} doc={doc[:12]} chat={chat[:12]}")
        print("  接着跑 setup_models.py 注册模型，再跑 reparse.py 解析。")
        return 0

    code, j = c._req("GET", "/api/v1/datasets", headers=c._h())
    for d in j.get("data", []):
        print(f"  {d['name']}  chunk_method={d['chunk_method']}  "
              f"chunks={d['chunk_count']}  docs={d['document_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
