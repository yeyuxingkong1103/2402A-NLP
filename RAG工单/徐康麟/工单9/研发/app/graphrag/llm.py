# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 来源说明：本文件复用自工单7/工单8（上游只读基座），工单9 未修改其业务逻辑，
#           仅补齐工单9 的编号头注释以满足交付纪律（_共享约定.md §4.2）。
"""工单8 Ollama 客户端：答案生成 + 图谱关系抽样补抽（统统带超时与显式降级）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

设计纪律（与 工单7 的 ``llm_client`` 同源，但按 Graph RAG 场景精简）：
    * 只用 ``requests``（已装），**不引入新依赖**；
    * 每次调用都有硬超时（``GRAPHRAG_LLM__TIMEOUT_S``），网络/模型缺失一律返回 ``None`` 由上层降级；
    * 生成提示词强制「只依据证据作答、给出块号」，不允许模型自由发挥；
    * 关系补抽提示词要求严格 JSON 数组，解析失败即丢弃该块的补抽结果（不修补、不猜）。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

#: 允许的规范关系名（写进提示词，避免模型自由造关系）
ALLOWED_RELATIONS: tuple[str, ...] = (
    "发行", "披露", "持股", "任职", "归属于", "同比变动", "位于", "母子公司", "拥有", "指标取值",
)

ANSWER_PROMPT = """你是严谨的金融年报问答助手。请**只依据给定的证据块**回答用户问题。

硬性要求：
1. 只使用下面【证据块】里的信息，**不得**引入任何外部知识或猜测；证据不足时直接回答「不清楚」。
2. 每个结论后用块号标注来源，形如 `[块3]`；数字必须与原文完全一致（含千分位与单位）。
3. 答案控制在 220 字以内，先给结论再给关键数据；中文提问用中文回答，英文提问用英文回答。
4. 不要输出「根据证据」这类套话，直接给答案。

【证据块】
{evidence}

【用户问题】
{question}

【答案】
"""

RELATION_PROMPT = """从下面的年报文本片段中抽取金融实体关系。只允许使用这些关系名：{relations}。

规则：
1. 只输出 JSON 数组，不要任何解释、不要 markdown 代码块。
2. 每项形如 {{"head":"实体A","relation":"关系名","tail":"实体B","evidence":"原文片段"}}。
3. 实体名必须是文本里**原样出现**的词（公司名/指标名/金额/比例/日期/产品/人名），不得改写或翻译。
4. 最多输出 {limit} 条；没有可抽的就输出 []。

文本片段：
{text}

JSON："""


@dataclass(slots=True)
class LLMResult:
    """一次生成的结果（含实测耗时，供报告与截图引用真实数字）。"""

    text: str
    model: str
    elapsed_ms: float
    ok: bool = True
    error: str | None = None
    prompt_chars: int = 0
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "model": self.model, "elapsed_ms": round(self.elapsed_ms, 2),
                "chars": len(self.text or ""), "error": self.error, "prompt_chars": self.prompt_chars,
                "attempts": self.attempts}


class GraphLLM:
    """Ollama HTTP 客户端（生成 + 关系补抽）。"""

    def __init__(self, cfg: Any, *, logger: Any = None) -> None:
        self.cfg = cfg
        self.log = logger
        settings: Mapping[str, Any] = getattr(cfg, "llm", {}) or {}
        self.base_url = str(settings.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
        self.model = str(settings.get("model") or "qwen2.5:3b")
        self.timeout_s = float(settings.get("timeout_s") or 45.0)
        self.num_ctx = int(settings.get("num_ctx") or 2048)
        #: Ollama 模型驻留策略：默认 ``"0s"`` = 用完立刻卸载（资源纪律，见 config 注释）
        self.keep_alive = str(settings.get("keep_alive") or "0s")
        self._available: bool | None = None

    # -- 可用性探测 --------------------------------------------------------
    def available(self, *, force: bool = False) -> bool:
        """探测 Ollama 是否可用（结果缓存；失败只 WARN，不抛异常）。"""
        if self._available is not None and not force:
            return self._available
        try:
            import requests

            resp = requests.get(f"{self.base_url}/api/tags", timeout=2.5)
            ok = resp.status_code == 200
            if ok:
                names = [str(m.get("name") or "") for m in (resp.json().get("models") or [])]
                if names and not any(n.split(":")[0] == self.model.split(":")[0] for n in names):
                    if self.log is not None:
                        self.log.log_event("llm.model_missing", level="WARNING", want=self.model,
                                           have=names[:20], action="生成层降级为抽取式")
                    ok = False
            self._available = ok
        except Exception as exc:  # noqa: BLE001 —— 无 Ollama 时必须降级而不是崩
            self._available = False
            if self.log is not None:
                self.log.log_event("llm.probe_failed", level="WARNING",
                                   error=f"{type(exc).__name__}: {exc}", base_url=self.base_url,
                                   action="生成层降级为抽取式")
        return bool(self._available)

    # -- 生成 -------------------------------------------------------------
    def _chat(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512,
              timeout_s: float | None = None, think: bool = False) -> LLMResult:
        import requests

        started = time.perf_counter()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # ⚠️ 资源纪律（`_共享约定.md` §5.5，事故后禁令）：每次请求都带 keep_alive="0s"，
            #    用完立刻卸载模型 —— 禁止 qwen2.5:3b 与 bge-m3 两个模型同时驻留
            #    （实测两模型同时驻留约 2.8 GB，曾把系统提交内存推到 96% 触发 OOM）。
            "keep_alive": str(self.keep_alive),
            "options": {"temperature": float(temperature), "num_predict": int(max_tokens),
                        "num_ctx": int(self.num_ctx)},
        }
        if not think:
            payload["think"] = False
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload,
                                 timeout=float(timeout_s or self.timeout_s))
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            if resp.status_code != 200:
                return LLMResult("", self.model, elapsed, ok=False,
                                 error=f"HTTP {resp.status_code}: {resp.text[:160]}",
                                 prompt_chars=len(prompt))
            data = resp.json()
            text = str(data.get("response") or "").strip()
            return LLMResult(text, self.model, elapsed, ok=bool(text), error=None if text else "空响应",
                             prompt_chars=len(prompt))
        except Exception as exc:  # noqa: BLE001 —— 超时/连接失败都要能被上层识别
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            return LLMResult("", self.model, elapsed, ok=False,
                             error=f"{type(exc).__name__}: {exc}", prompt_chars=len(prompt))

    def answer(self, question: str, evidence_blocks: Sequence[Mapping[str, Any]], *,
               language: str = "zh", max_tokens: int = 400) -> LLMResult:
        """基于证据块生成答案（证据为空 ⇒ 直接返回 ok=False，不浪费一次推理）。"""
        if not evidence_blocks:
            return LLMResult("", self.model, 0.0, ok=False, error="无证据块", prompt_chars=0)
        prompt = ANSWER_PROMPT.replace("{evidence}", render_evidence(evidence_blocks)) \
                             .replace("{question}", str(question or ""))
        if language == "en":
            prompt = prompt.replace("中文提问用中文回答，英文提问用英文回答", "请用英文回答（English）")
        result = self._chat(prompt, temperature=0.2, max_tokens=max_tokens)
        if self.log is not None:
            self.log.log_event("llm.answer", ok=result.ok, model=result.model,
                               elapsed_ms=result.elapsed_ms, prompt_chars=result.prompt_chars,
                               error=result.error)
        return result

    # -- 关系补抽 ---------------------------------------------------------
    def extract_relations(self, text: str, *, chunk_id: str = "", limit: int = 3) -> list[dict[str, Any]]:
        """对单块做关系补抽；返回 ``[{head, relation, tail, evidence}]``（解析失败返回 []）。"""
        prompt = (RELATION_PROMPT
                  .replace("{relations}", "、".join(ALLOWED_RELATIONS))
                  .replace("{limit}", str(int(limit)))
                  .replace("{text}", str(text or "")[:400]))
        result = self._chat(prompt, temperature=0.0, max_tokens=320, think=False)
        if not result.ok:
            if self.log is not None:
                self.log.log_event("llm.relation_failed", level="WARNING", chunk_id=chunk_id,
                                   error=result.error, elapsed_ms=result.elapsed_ms)
            return []
        rows = parse_relations_json(result.text, allowed=ALLOWED_RELATIONS, limit=limit)
        for row in rows:
            row["chunk_id"] = chunk_id
            row["elapsed_ms"] = result.elapsed_ms
        return rows


def render_evidence(blocks: Sequence[Mapping[str, Any]], *, max_chars: int = 420) -> str:
    """把证据块渲染成提示词用的编号列表（块号与最终返回的 chunk 顺序一致）。"""
    lines: list[str] = []
    for idx, block in enumerate(blocks, 1):
        text = re.sub(r"\s+", " ", str(block.get("content") or block.get("text") or "")).strip()
        page = block.get("page")
        file_name = str(block.get("file_name") or "")
        source = f"{file_name} 第{page}页" if page else file_name
        lines.append(f"[块{idx}]（来源：{source}；chunk_id={block.get('chunk_id')}）\n{text[:max_chars]}")
    return "\n\n".join(lines)


_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def parse_relations_json(text: str, *, allowed: Sequence[str] = ALLOWED_RELATIONS,
                         limit: int = 3) -> list[dict[str, Any]]:
    """从模型输出里稳健地取出 JSON 数组（兼容 ```json 围栏与前后废话）。"""
    raw = str(text or "").strip()
    if not raw:
        return []
    raw = raw.replace("```json", "```").replace("```", "")
    match = _JSON_ARRAY.search(raw)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except ValueError:
        # 常见毛病：尾随逗号 / 单引号 —— 只做最小修复，修不好就放弃（不猜）
        candidate = match.group(0).replace("'", '"')
        candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
        try:
            data = json.loads(candidate)
        except ValueError:
            return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        head = str(item.get("head") or "").strip()
        relation = str(item.get("relation") or "").strip()
        tail = str(item.get("tail") or "").strip()
        if not head or not tail or relation not in set(allowed):
            continue
        out.append({"head": head, "relation": relation, "tail": tail,
                    "evidence": str(item.get("evidence") or "").strip()[:200]})
        if len(out) >= int(limit):
            break
    return out
