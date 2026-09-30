# -*- coding: utf-8 -*-
"""入库前的「这份资料是否符合该角色的设定」判定。

为什么要有它
------------
知识按角色分区（见 `docs/ARCHITECTURE.md` §12）：律师的法条不能被医生角色检索到。
但**上传时谁能保证选对了角色**？选错就等于把知识塞进了错误的抽屉，而且检索侧
严格过滤会让它「谁都搜不到」或「被不该看到的角色看到」。所以入库前先判一次，
把结论摊给用户看，**由用户在「当前角色 / 建议角色」之间做决定**。

三档结论（缺一不可，尤其是第三档）
----------------------------------
* ``fit=True``   相符 -> 直接入当前角色；
* ``fit=False``  不符 -> 给出 ``suggested_role``（建议归到哪个角色）+ 理由；
* ``fit=None``   **无法判定**（裁判不可用/调用失败）-> 明确说"判不了"，
  **绝不默认放行、绝不假装通过**，让用户自己决定。

判定由两部分组成
----------------
1. **确定性预筛**（免费、快、可解释，永远有结果）：领域关键词命中统计 +（若该角色已有知识）
   与该角色知识库的向量相似度。它给出 ``signals``，是裁判结论的旁证，也是 LLM 不可用时的提示。
2. **LLM 裁判**（默认 DeepSeek，可切本地 Ollama / 将来的 Qwen3.8-27B）：按严格 JSON 出结论。

配置（环境变量；本模块直接读环境变量是**有意**的 —— `config.py` 已冻结，
而判定属于服务层策略，改这里不影响核心链路）
------------------------------------------------
* ``FIT_CHECK_PROVIDER``  ``auto``（默认）| ``deepseek`` | ``ollama`` | ``none``
  ``auto`` = 有 ``DEEPSEEK_API_KEY`` 就用 deepseek，否则用 ollama，都没有 -> 无法判定
* ``FIT_CHECK_MODEL``     默认：deepseek -> ``deepseek-chat``；ollama -> ``OLLAMA_MODEL``
* ``FIT_CHECK_SAMPLE_CHARS`` 送给裁判的正文节选上限（默认 3000）
* ``FIT_CHECK_MAX_TOKENS``   裁判回答上限（默认 300）
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any

from ..roles import ROLE_LIBRARY, get_role

logger = logging.getLogger(__name__)

#: 每个角色的领域关键词（**确定性预筛**用；宁可粗一点，也别假装精确）
ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "lawyer": ("法律", "法条", "条例", "规定", "办法", "判决", "裁定", "诉讼", "仲裁", "合同",
               "刑法", "民法", "劳动法", "司法解释", "人民法院", "人民检察院", "违法", "侵权"),
    "doctor": ("疾病", "症状", "诊断", "治疗", "用药", "药物", "临床", "患者", "手术", "医学",
               "病理", "检验", "指南", "并发症"),
    "tcm": ("中医", "辨证", "经络", "针灸", "方剂", "气血", "脏腑", "体质", "本草", "穴位"),
    "psychologist": ("心理", "情绪", "焦虑", "抑郁", "咨询", "认知", "依恋", "创伤", "CBT"),
    "financial_planner": ("理财", "资产配置", "基金", "保险", "收益率", "风险偏好", "养老规划",
                          "财富", "复利"),
    "stock_advisor": ("股票", "证券", "行情", "指数", "市盈率", "K线", "板块", "持仓", "估值"),
}

_JUDGE_SYSTEM = (
    "你是「角色知识归属审核员」。你的任务是判断给定资料是否属于某个角色的知识范围，"
    "并在不属于时从候选角色里挑一个最合适的。"
    "只输出一个 JSON 对象，不要任何解释文字、不要 markdown 代码块。"
)

#: 固定种子：判定必须**可复现**（F-D 的现象是同一份《民法典》文本 3 次得到
#: false/true/false，且 reason 与结论互相矛盾）。temperature=0 之外再固定 seed，
#: 让"同一输入 → 同一结论"成为硬约束。
FIT_CHECK_SEED = 20260916

#: 固定 rubric（写进 system 提示词）：把三档判定规则讲成可执行的三条，
#: 减少"结论与理由各说一套"的余地。
_JUDGE_RUBRIC = (
    "判定规则（固定 rubric，按顺序执行）：\n"
    "1) 资料主题与「待判定角色」的领域一致 → fit=true；\n"
    "2) fit=false **必须同时**给出 suggested_role（从候选清单里选一个），"
    "给不出就输出 fit=null（无法判定），不要硬报 false；\n"
    "3) 证据不足 / 无法判断范围 → fit=null；confidence 反映你的确信程度。"
)


@dataclass
class FitVerdict:
    """判定结果（直接作为接口响应体，字段保持稳定）。"""
    fit: bool | None = None
    confidence: float = 0.0
    reason: str = ""
    suggested_role: str | None = None
    suggested_role_name: str = ""
    judge: str = "none"                       # deepseek | ollama | none
    judge_model: str = ""
    degraded: bool = True                     # True = 裁判不可用，结论仅供参考
    error: str = ""
    signals: dict[str, Any] = field(default_factory=dict)
    sample_chars: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# 确定性预筛
# --------------------------------------------------------------------------

def keyword_signals(text: str, role_id: str) -> dict:
    """领域关键词命中统计（纯本地、可解释）。"""
    sample = text[:20000]
    hits = {rid: sum(sample.count(kw) for kw in kws) for rid, kws in ROLE_KEYWORDS.items()}
    target_hits = hits.get(role_id, 0)
    ranked = sorted(hits.items(), key=lambda kv: -kv[1])
    top_role, top_hits = ranked[0] if ranked else ("", 0)
    return {
        "target_role_hits": target_hits,
        "top_role": top_role,
        "top_role_hits": top_hits,
        "keyword_hint_matches": bool(top_role == role_id and top_hits > 0),
        "top_roles": [{"role_id": r, "hits": h} for r, h in ranked[:4] if h > 0],
    }


def kb_similarity_signals(engine, role_id: str, text: str) -> dict:
    """与该角色**已有知识**的向量相似度（角色库为空时给 null，不编数）。"""
    try:
        if engine is None or engine.store.count() == 0:
            return {"role_kb_chunks": 0, "top1_similarity": None}
        scope = {"is_parent": False, "role_id": role_id}
        chunks = engine.store.all_chunks(scope)
        if not chunks:
            return {"role_kb_chunks": 0, "top1_similarity": None}
        vector = engine.embedder.embed_query(text[:1000])
        hits = engine.store.search(vector, top_k=1, where=scope)
        top1 = round(float(hits[0].score), 4) if hits else None
        return {"role_kb_chunks": len(chunks), "top1_similarity": top1}
    except Exception as exc:  # noqa: BLE001 - 预筛失败不影响判定主流程
        logger.warning("知识库相似度预筛失败：%s: %s", type(exc).__name__, exc)
        return {"role_kb_chunks": None, "top1_similarity": None,
                "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# LLM 裁判
# --------------------------------------------------------------------------

class RoleFitJudge:
    def __init__(self, config=None, engine=None) -> None:
        self.config = config
        self.engine = engine
        self.provider = (os.environ.get("FIT_CHECK_PROVIDER") or "auto").strip().lower()
        self.sample_chars = int(os.environ.get("FIT_CHECK_SAMPLE_CHARS") or 3000)
        self.max_tokens = int(os.environ.get("FIT_CHECK_MAX_TOKENS") or 300)
        self._client = None
        self._client_label = ""
        self._client_model = ""

    # ---------- 选裁判 ----------
    def _build_client(self):
        if self._client is not None or self._client_label == "none":
            return self._client
        from ..generate.llm_base import build_llm_client

        want = self.provider
        has_deepseek = bool(os.environ.get("DEEPSEEK_API_KEY"))
        if want == "auto":
            want = "deepseek" if has_deepseek else "ollama"
        if want == "deepseek" and not has_deepseek:
            self._client_label = "none"
            logger.warning("FIT_CHECK_PROVIDER=deepseek 但没有 DEEPSEEK_API_KEY -> 无法判定")
            return None
        try:
            if want == "deepseek":
                model = os.environ.get("FIT_CHECK_MODEL") or "deepseek-chat"
                self._client = build_llm_client("deepseek", model=model,
                                                temperature=0.0, max_tokens=self.max_tokens)
            elif want == "ollama":
                model = (os.environ.get("FIT_CHECK_MODEL")
                         or getattr(getattr(self.config, "ollama", None), "llm_model", "")
                         or "qwen2.5:3b")
                self._client = build_llm_client(                    "ollama", model=model, temperature=0.0, max_tokens=self.max_tokens,
                    ollama_base_url=getattr(getattr(self.config, "ollama", None), "base_url", ""),
                    timeout=90.0)
            else:
                self._client_label = "none"
                return None
            self._client_label = want
            self._client_model = getattr(self._client, "model", "")
            # 显式固定 seed（F-D）：判定必须**可复现** —— 同一份资料 3 次必须给同一结论。
            # 只靠 temperature=0 不够（采样器/并发内核仍可能选到等价分支）；Ollama 客户端
            # 会把它写进 options.seed，其它 provider 忽略未知属性即可。
            if self._client is not None:
                try:
                    self._client.seed = FIT_CHECK_SEED
                except Exception:  # noqa: BLE001 - 设不上也不该让判定不可用
                    logger.debug("裁判客户端不支持 seed，仅依赖 temperature=0", exc_info=True)
            logger.info("判定裁判已就绪：provider=%s model=%s seed=%s", self._client_label,
                        self._client_model, FIT_CHECK_SEED)
            return self._client
        except Exception as exc:  # noqa: BLE001 - 造不出来就明确"无法判定"
            logger.exception("判定裁判初始化失败（provider=%s）", want)
            self._client_label = "none"
            return None

    def provider_status(self) -> dict:
        """给前端/运维看：当前裁判是谁、能不能用。"""
        client = self._build_client()
        return {
            "requested": self.provider,
            "judge": self._client_label,
            "model": self._client_model,
            "available": client is not None,
            "deepseek_key_present": bool(os.environ.get("DEEPSEEK_API_KEY")),
        }

    # ---------- 提示词 ----------
    def _messages(self, role_id: str, filename: str, sample: str) -> list[dict]:
        role = get_role(role_id)
        catalogue = "\n".join(
            f"- {r.role_id}｜{r.name}｜领域：{r.domain}" for r in ROLE_LIBRARY.values())
        user = (
            f"【待判定角色】\nrole_id: {role.role_id}\n名称: {role.name}\n"
            f"领域: {role.domain}\n人设: {role.persona}\n\n"
            f"【候选角色清单】（只能从这里选 role_id）\n{catalogue}\n\n"
            f"【资料文件名】{filename}\n"
            f"【资料正文节选】\n\"\"\"\n{sample}\n\"\"\"\n\n"
            "请判断：这份资料是否属于上述「待判定角色」的知识范围？\n"
            "只输出 JSON：{\"fit\": true/false, \"confidence\": 0 到 1 的小数, "
            "\"reason\": \"不超过 60 字的中文理由\", \"suggested_role\": \"role_id 或 null\"}\n"
            "注意：suggested_role 仅在 fit=false 时给出；不确定就给 null。"
        )
        return [{"role": "system", "content": _JUDGE_SYSTEM + "\n" + _JUDGE_RUBRIC},
                {"role": "user", "content": user}]

    @staticmethod
    def _parse(text: str) -> dict:
        """从模型输出里抠出 JSON（容忍 ```json 代码块与前后废话）。"""
        raw = (text or "").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001 - 退化为"找第一个平衡的 {...}"
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise ValueError(f"模型输出里没有 JSON：{raw[:200]!r}")
            return json.loads(match.group(0))

    # ---------- 主入口 ----------
    def judge(self, text: str, role_id: str, *, filename: str = "") -> FitVerdict:
        sample = (text or "")[: self.sample_chars]
        verdict = FitVerdict(sample_chars=len(sample))
        signals = {"keywords": keyword_signals(text or "", role_id)}
        signals.update(kb_similarity_signals(self.engine, role_id, text or ""))
        verdict.signals = signals

        # 文本抽不出来（扫描件/空文件）时**不要**去调模型：没有依据的判定等于编，
        # 如实报"无法判定"，让用户自己决定。
        if len((text or "").strip()) < 50:
            verdict.fit = None
            verdict.degraded = True
            verdict.judge = "none"
            verdict.reason = (f"文件里只抽到 {len((text or '').strip())} 个字符，"
                              "没有足够依据判定（可能是扫描件或空文件）")
            hint = signals["keywords"].get("top_role")
            if hint and hint != role_id and signals["keywords"].get("top_role_hits"):
                verdict.suggested_role = hint
                verdict.suggested_role_name = get_role(hint).name
            return verdict

        client = self._build_client()
        if client is None:
            # 关键：判不了就说判不了，并把确定性信号作为"提示"给出，绝不默认放行
            hint = signals["keywords"].get("top_role")
            verdict.degraded = True
            verdict.judge = "none"
            verdict.reason = ("判定模型不可用（未配置 DEEPSEEK_API_KEY，Ollama 也不可用）；"
                              "这只是一份未经验证的资料")
            if hint and hint != role_id:
                verdict.suggested_role = hint
                verdict.suggested_role_name = get_role(hint).name
                verdict.reason += f"；按领域关键词更像是「{verdict.suggested_role_name}」的资料"
            return verdict

        verdict.judge = self._client_label
        verdict.judge_model = self._client_model
        try:
            output = client.chat(self._messages(role_id, filename, sample))
            data = self._parse(output)
            fit = bool(data.get("fit"))
            suggested = data.get("suggested_role") or None
            if isinstance(suggested, str) and suggested.strip().lower() in ("null", "none", ""):
                suggested = None
            if suggested and suggested not in ROLE_LIBRARY:
                logger.warning("裁判给的建议角色不在角色库里，忽略：%r", suggested)
                suggested = None
            # 一致性收口（F-D）：rubric 第 2 条要求 fit=false 必须配 suggested_role。
            # 模型偶尔给出"结论说不符、却指不出该归哪里"的自相矛盾输出 —— 与其把一个
            # 站不住的 false 交给用户（他可能据此把资料塞进错误的抽屉），不如如实说
            # 「无法判定」（fit=None），让用户自己决定。**绝不放行成 true**。
            if not fit and not suggested:
                logger.warning("裁判输出自相矛盾（fit=false 但无 suggested_role）→ 按无法判定处理")
                verdict.fit = None
                verdict.degraded = True
                verdict.confidence = round(float(data.get("confidence") or 0.0), 3)
                verdict.reason = (str(data.get("reason") or "").strip()[:160]
                                  or "判定模型只给了结论、没给出建议归属") + \
                    "（结论与理由不一致，按「无法判定」处理，请自行决定）"
                verdict.suggested_role = None
                verdict.suggested_role_name = ""
                return verdict
            verdict.fit = fit
            verdict.confidence = round(float(data.get("confidence") or 0.0), 3)
            verdict.reason = str(data.get("reason") or "").strip()[:200]
            verdict.suggested_role = suggested
            verdict.suggested_role_name = get_role(suggested).name if suggested else ""
            verdict.degraded = False
            logger.info("判定完成：role=%s fit=%s conf=%s suggested=%s",
                        role_id, verdict.fit, verdict.confidence, verdict.suggested_role)
        except Exception as exc:  # noqa: BLE001 - 判定失败必须显式降级，不能默认通过
            logger.exception("LLM 判定失败（role=%s）", role_id)
            verdict.degraded = True
            verdict.fit = None
            verdict.error = f"{type(exc).__name__}: {exc}"
            verdict.reason = f"判定模型调用失败（{verdict.error}），无法给出结论"
        return verdict
