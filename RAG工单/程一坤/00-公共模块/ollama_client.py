# -*- coding: utf-8 -*-
"""
Ollama 客户端封装
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
说明：统一封装 LLM 对话生成 / 文本向量化，带重试与超时控制，
     并强制绕过系统代理（本机代理会劫持 127.0.0.1 请求导致连接失败）。
"""
import os       # 读取/删除代理相关环境变量
import time     # 重试之间的退避等待
import json     # 解析 Ollama 返回的 JSON 响应体
import requests  # HTTP 客户端，直接调用 Ollama 的 REST API

# 关键：清除代理环境变量，否则 127.0.0.1 请求会被 http_proxy 劫持
# （沙箱/公司网络常设 http_proxy，requests 默认会走代理转发本地请求导致连接失败）
# 注意：必须在 import config / 创建 Session 之前清除才能生效
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)  # pop 带默认值 None，键不存在时不报错

# 导入公共配置：服务地址与各模型名均可在 config 中通过环境变量覆盖
from config import OLLAMA_BASE, LLM_MODEL, EMBED_MODEL, EMBED_BATCH, VL_MODEL


class OllamaClient:
    """Ollama HTTP API 客户端（含重试机制）"""

    def __init__(self, base_url=None, llm_model=None, embed_model=None, vl_model=None):
        # 各参数允许外部覆盖，缺省回落到 config.py 中的全局配置
        self.base = base_url or OLLAMA_BASE
        self.llm_model = llm_model or LLM_MODEL      # 生成模型（qwen2.5:7b-instruct）
        self.embed_model = embed_model or EMBED_MODEL  # 向量模型（bge-m3，1024维）
        self.vl_model = vl_model or VL_MODEL         # 多模态模型（工单04 图片理解用）
        self.session = requests.Session()  # 复用 TCP 连接，比每次 requests.post 快
        self.session.trust_env = False  # 不读取系统代理

    # ── 健康检查 ───────────────────────────────────────────
    def is_alive(self):
        """探测 Ollama 服务是否存活：GET /api/tags 列出已安装模型"""
        try:
            # 超时设 3 秒：健康检查必须快速失败，不能阻塞主流程
            r = self.session.get(f"{self.base}/api/tags", timeout=3)
            return r.status_code == 200  # 200 表示服务正常响应
        except Exception:
            # 网络不通/服务未启动等异常统一视为"不存活"，不向外抛
            return False

    # ── LLM 生成 ───────────────────────────────────────────
    def chat(self, messages, temperature=0.1, num_predict=1024, retries=3, timeout=120):
        """调用 /api/chat 生成回答。messages: [{"role","content"}, ...]"""
        # 组装请求体：temperature=0.1 低温度保证问答输出稳定不发散；
        # num_predict 限制最大生成 token 数，防止长篇输出拖慢响应
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,  # 关闭流式：一次性拿完整回答，便于解析与计时
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        last_err = None  # 记录最后一次异常，重试耗尽后用于报错定位
        for i in range(retries):
            try:
                # json=payload 自动序列化并设置 Content-Type；timeout 防止模型卡死挂起请求
                r = self.session.post(f"{self.base}/api/chat", json=payload, timeout=timeout)
                r.raise_for_status()  # 非 2xx 状态码（如模型未加载 404/500）抛异常进入重试
                # Ollama 的回答固定在 message.content 字段；strip() 去掉首尾空白
                return r.json()["message"]["content"].strip()
            except Exception as e:
                last_err = e
                time.sleep(2 * (i + 1))  # 退避重试
        # 重试全部失败：抛出带原因的异常，由上层决定如何处理
        raise RuntimeError(f"LLM 调用失败(重试{retries}次): {last_err}")

    def generate(self, prompt, **kw):
        """便捷方法：单条 prompt 生成"""
        # 把单条字符串包装成 chat 接口要求的 messages 列表格式
        return self.chat([{"role": "user", "content": prompt}], **kw)

    # ── 多模态视觉理解（工单04） ───────────────────────────
    def vl_describe(self, image_path, prompt, retries=2, timeout=600):
        """用多模态模型理解图片内容，返回描述文本"""
        import base64  # 局部导入：仅本方法需要，避免模块加载开销
        with open(image_path, "rb") as f:  # 必须二进制模式读图，文本模式会损坏图片数据
            # Ollama 要求图片以 base64 字符串内嵌在请求中传输
            b64 = base64.b64encode(f.read()).decode()
        # 视觉模型同样走 /api/chat，区别是把 base64 图片放进消息的 images 字段
        payload = {
            "model": self.vl_model,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 500},  # 图片描述无需长输出
        }
        last_err = None
        for i in range(retries):
            try:
                # 视觉模型推理慢（需先"看图"），timeout 放宽到 600 秒
                r = self.session.post(f"{self.base}/api/chat", json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json()["message"]["content"].strip()
            except Exception as e:
                last_err = e
                time.sleep(3 * (i + 1))  # 线性退避：第1次等3s、第2次等6s
        raise RuntimeError(f"视觉模型调用失败: {last_err}")

    # ── 向量化 ─────────────────────────────────────────────
    def embed(self, text, retries=3):
        """单条文本向量化，返回 list[float]"""
        last_err = None
        for i in range(retries):
            try:
                # 调 /api/embeddings 接口；text 截断到前 4000 字符，
                # 防止超长文本超出 bge-m3 上下文导致 Ollama 报 500
                r = self.session.post(
                    f"{self.base}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text[:4000]},
                    timeout=60,
                )
                r.raise_for_status()
                return r.json()["embedding"]  # 返回 1024 维浮点向量（bge-m3）
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (i + 1))  # 退避等待后重试
        raise RuntimeError(f"向量化失败: {last_err}")

    def embed_batch(self, texts, batch_size=None, progress=True):
        """批量向量化（顺序执行，带进度显示）"""
        batch_size = batch_size or EMBED_BATCH  # 未指定时用 config 中的默认批大小 32
        vecs = []
        for i, t in enumerate(texts):
            vecs.append(self.embed(t))  # Ollama 无批量接口，逐条调用；顺序执行避免并发打爆服务
            if progress and (i + 1) % batch_size == 0:
                # 每完成一批打印一次进度，千级分块时让用户知道程序没有卡死
                print(f"  向量化进度: {i+1}/{len(texts)}")
        return vecs


# 模块级单例，供各工单直接导入使用
# 全局只建一个 Session，避免每个工单脚本重复创建连接池
client = OllamaClient()
