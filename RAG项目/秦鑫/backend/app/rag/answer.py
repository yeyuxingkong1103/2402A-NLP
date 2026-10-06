import io
import logging
from pathlib import Path
import re
from time import perf_counter

from PIL import Image, ImageDraw, ImageFont

from ..core import PUBLIC_COLLECTION_LABELS
from ..models import ModelGateway, build_answer_messages, build_no_evidence_answer_messages, build_solution_messages, extract_final_answer, validate_final_answer


logger = logging.getLogger("law_rag.answer")

SOURCE_LABELS = {
    "public": "公共法律库",
    "private": "用户上传材料",
    "user_upload": "用户上传材料",
    "web": "网页资料",
}


def model_setting_value(model: ModelGateway, name: str, default: int) -> int:
    value = getattr(getattr(model, "settings", None), name, default)
    try:
        return max(1, int(value or default))
    except (TypeError, ValueError):
        return default


def model_setting_float(model: ModelGateway, name: str, default: float) -> float:
    value = getattr(getattr(model, "settings", None), name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_citations(rows: list[dict]) -> list[dict]:
    return [
        {
            "index": i,
            "title": row.get("title", "未知来源"),
            "source_type": row.get("source_type", "unknown"),
            "source_label": row.get("source_label") or (PUBLIC_COLLECTION_LABELS.get(row.get("collection", ""), "公共法律库") if row.get("source_type") == "public" else SOURCE_LABELS.get(row.get("source_type", "unknown"), "未知来源")),
            "source_id": row.get("source_id", ""),
            "collection": row.get("collection", ""),
            "file_name": row.get("file_name", ""),
            "url": row.get("url") or row.get("link") or (row.get("source_id", "") if row.get("source_type") == "web" and str(row.get("source_id", "")).startswith(("http://", "https://")) else ""),
        }
        for i, row in enumerate(rows, start=1)
    ]


class AnswerGenerator:
    def __init__(self, model: ModelGateway):
        self.model = model

    @staticmethod
    def _looks_like_signature_type_error(error: TypeError) -> bool:
        message = str(error)
        markers = (
            "unexpected keyword argument",
            "got an unexpected keyword",
            "takes no keyword arguments",
            "positional argument",
            "required positional argument",
            "missing 1 required",
        )
        return any(marker in message for marker in markers)

    @staticmethod
    def _call_model(func, messages: list[dict], thinking_enabled: bool | None = None, max_tokens: int | None = None, temperature: float | None = None):
        try:
            if max_tokens is not None:
                return func(messages, max_tokens=max_tokens, thinking_enabled=thinking_enabled, temperature=temperature)
            return func(messages, thinking_enabled=thinking_enabled, temperature=temperature)
        except TypeError as exc:
            if not AnswerGenerator._looks_like_signature_type_error(exc):
                raise
            try:
                if max_tokens is not None:
                    return func(messages, max_tokens=max_tokens)
                return func(messages)
            except TypeError as retry_exc:
                if not AnswerGenerator._looks_like_signature_type_error(retry_exc):
                    raise
                return func(messages)

    @staticmethod
    def _clean_model_output(text: str, citations: list[dict] | None = None) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        if value.startswith("```"):
            lines = value.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            value = "\n".join(lines).strip()
        valid, issues = validate_final_answer(value, citations)
        if valid:
            return extract_final_answer(value)
        if "<final_answer>" in value or "</final_answer>" in value:
            logger.warning(
                "模型输出 final_answer 标签不完整，按内容边界清洗",
                extra={"event": "final_answer_envelope_repaired", "fields": {"issues": issues[:4]}},
            )
            if "<final_answer>" in value:
                value = value.split("<final_answer>", 1)[1]
            if "</final_answer>" in value:
                value = value.split("</final_answer>", 1)[0]
        return value.strip()

    @staticmethod
    def _source_details(row: dict) -> str:
        details = []
        source_id = str(row.get("source_id") or "").strip()
        if source_id:
            details.append(f"来源ID={source_id}")
        article_label = ""
        value = row.get("article_number")
        if value not in (None, ""):
            article_label = f"第{value}条"
        else:
            source_id_value = str(row.get("source_id") or "")
            collection = str(row.get("collection") or "")
            import re

            if collection == "civil_code_articles":
                match = re.search(r"article_(\d+)$", source_id_value)
                if match:
                    article_label = f"第{match.group(1)}条"
            elif collection == "civil_elements":
                match = re.search(r":(\d+)$", source_id_value)
                if match:
                    article_label = f"第{match.group(1)}条"
        if article_label:
            details.append(f"条号={article_label}")
        field_labels = (
            ("law_name", "法律或文件"),
            ("document_number", "文号"),
            ("part_name", "章节"),
            ("cause_of_action", "案由"),
            ("case_number", "案号"),
            ("core_legal_issue", "争议焦点"),
            ("rule_name", "规则名称"),
            ("rule_summary", "规则摘要"),
            ("question", "问答题目"),
            ("source_case_number", "来源案号"),
            ("target_id", "目标来源"),
        )
        for field, label in field_labels:
            value = row.get(field)
            if value in (None, "", []):
                continue
            if isinstance(value, list):
                value = "、".join(str(item) for item in value[:5])
            details.append(f"{label}={value}")
        return "；".join(details)

    @staticmethod
    def _source_highlight_terms(row: dict) -> list[str]:
        if row.get("source_type") != "public" or not row.get("collection"):
            return []
        fields = ("title", "document_number", "cause_of_action", "rule_name", "question")
        terms = []
        article_label = ""
        value = row.get("article_number")
        if value not in (None, ""):
            article_label = f"第{value}条"
        else:
            source_id = str(row.get("source_id") or "")
            collection = str(row.get("collection") or "")
            import re

            if collection == "civil_code_articles":
                match = re.search(r"article_(\d+)$", source_id)
                if match:
                    article_label = f"第{match.group(1)}条"
            elif collection == "civil_elements":
                match = re.search(r":(\d+)$", source_id)
                if match:
                    article_label = f"第{match.group(1)}条"
        if article_label:
            terms.append(article_label)
        for field in fields:
            value = row.get(field)
            if value in (None, "", []):
                continue
            if isinstance(value, list):
                values = [str(item).strip() for item in value if str(item).strip()]
            else:
                values = [str(value).strip()]
            for item in values:
                if 2 <= len(item) <= 140 and item not in terms:
                    terms.append(item)
        if row.get("collection") == "civil_cases":
            first_line = str(row.get("content") or "").strip().splitlines()[0:1]
            if first_line:
                snippet = first_line[0].strip()[:100]
                if len(snippet) >= 8 and snippet not in terms:
                    terms.append(snippet)
        return terms

    @staticmethod
    def ensure_source_highlights(answer: str, question: str, evidence: list[dict]) -> str:
        value = str(answer or "").strip()
        if not value or not evidence:
            return value
        terms = []
        question_text = str(question or "")
        for term in ("构成要件", "关键事实", "举证责任"):
            if term in question_text and term not in terms:
                terms.append(term)
        for row in evidence[:4]:
            for term in AnswerGenerator._source_highlight_terms(row):
                if term not in terms:
                    terms.append(term)
        missing = [term for term in terms if term and term not in value]
        if not missing:
            return value
        return f"{value}\n\n**依据核对**：{'；'.join(missing[:10])}。"

    @staticmethod
    def _evidence_context(evidence: list[dict], max_chars: int = 1200) -> str:
        parts = []
        for i, row in enumerate(evidence, start=1):
            label = row.get("source_label") or ("用户上传材料" if row.get("source_type") in {"private", "user_upload"} else "检索来源")
            details = AnswerGenerator._source_details(row)
            detail_text = f"\n关键来源信息：{details}" if details else ""
            parts.append(f"[{i}] 【{label}】{row.get('title')}{detail_text}\n{row.get('content', '')[:max_chars]}")
        return "\n\n".join(parts)

    def _fallback_answer(self, question: str, evidence: list[dict], citations: list[dict]) -> str:
        text = str(question or "").strip()
        evidence_question = any(marker in text for marker in ("证据", "材料", "证明", "举证"))
        labor_question = any(marker in text for marker in ("工资", "劳动", "加班", "辞退", "离职", "工伤"))
        source_note = f"本轮检索到 {len(citations)} 条参考来源，但来源只用于核对规则，不等同于已经证明你的具体事实。"
        if evidence_question and labor_question:
            actions = [
                "保存劳动合同、入职及离职材料、工资条或银行流水，证明劳动关系、工资标准和欠付金额。",
                "整理考勤、加班通知、工作记录、聊天记录和催款记录，并保留原始载体及完整上下文。",
                "制作工资明细和事实时间线，标出每一笔应付、实付、欠付及对应证据。",
            ]
            focus = "你的问题重点是准备证据，当前最有价值的是证明劳动关系、工资标准、实际出勤和欠付金额。"
        elif evidence_question:
            actions = [
                "先按时间顺序保存合同、付款、聊天、通知、照片视频和平台记录。",
                "为每项证据注明来源、形成时间、证明对象，并保留原始文件。",
                "补齐对方身份、争议经过、损失结果和你希望对方承担的责任。",
            ]
            focus = "你的问题重点是证据准备，关键不是材料越多越好，而是每项材料都能对应证明一个具体事实。"
        else:
            actions = [
                "先把事实按日期、人物、行为、结果整理成时间线。",
                "保存合同、付款、沟通、通知和其他原始材料，避免只保留截图。",
                "根据证据完整程度，再选择协商、书面催告、调解、仲裁或诉讼。",
            ]
            focus = "关于这个问题，目前可以先做方向性判断，但具体责任仍取决于事实和证据能否对应起来。"
        return (
            f"我知道你现在最想尽快确认该准备什么、下一步怎么走。关于“{text}”，先把重点放在可证明的事实和证据上。\n\n"
            f"## 核心判断\n**{focus}**\n\n"
            f"## 现在先做\n{chr(10).join(f'{i}. {item}' for i, item in enumerate(actions, 1))}\n\n"
            f"## 还需确认\n核对双方身份关系、关键时间、金额或损失、对方当前态度，以及是否已经存在仲裁、诉讼或其他期限风险。\n\n"
            f"## 依据说明\n{source_note} 引用内容应结合原始材料进一步核验。\n\n"
            "以上内容仅供法律信息参考；涉及较大金额、劳动仲裁、诉讼期限或重大权益，建议尽快让执业律师审核原始材料。"
        )

    def build_case_analysis(self, question: str, evidence: list[dict], answer: str, thinking_enabled: bool | None = None) -> list[str]:
        if not evidence:
            return [
                "实际争点：现在还缺少能直接对应到事实和法律规则的材料，暂时不能把结论说死。",
                "现有证据怎么看：当前没有检索到足够依据，最关键的是先补齐事实经过、时间节点、金额或标的、双方关系和已有凭证。",
                "法律上怎么判断：需要先确定属于哪类法律关系，再看请求权基础、举证责任和诉讼时效；没有证据时不能只凭主观描述作确定判断。",
                "接下来最实际的动作：把合同、聊天记录、付款凭证、通知函、法院文书或其他能证明关键事实的材料补充后，再继续追问具体处理路径。",
            ]
        context = self._evidence_context(evidence, max_chars=900)
        prompt = f"""
你是面向普通用户的法律助手。请基于用户问题、检索证据和正式答案，写一段可公开展示的“实际案情分析”。

用户问题：
{question}

检索证据：
{context}

正式答案：
{answer[:2400]}

要求：
- 必须真正围绕用户问题分析，不要写泛泛模板。
- 只输出 4 到 6 条要点，每条一行。
- 每条以这些标签之一开头：实际争点：、现有证据怎么看：、法律上怎么判断：、类似情况的处理倾向：、对方可能怎么反驳：、接下来最实际的动作：。
- 每条不少于 35 个中文字符，必须说明具体原因或判断逻辑。
- 可以引用来源编号或标题，但不要只罗列来源名称。
- 不要输出内部思考过程、自我纠错、草稿、模型推理链或标签外说明。
- 不要使用“事实是否成立、请求权基础是否明确、证据能否支持主张”这种固定套话，除非后面紧接着解释本案为什么。
"""
        try:
            try:
                text = self.model.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=900,
                    thinking_enabled=False,
                    temperature=model_setting_float(self.model, "answer_llm_temperature", 0.25),
                )
            except TypeError:
                text = self.model.chat([{"role": "user", "content": prompt}])
        except Exception:
            return self.rule_based_case_analysis(question, evidence, answer)
        lines = []
        for raw_line in str(text or "").replace("\r\n", "\n").split("\n"):
            line = raw_line.strip().lstrip("-•0123456789.、 ").strip()
            if not line:
                continue
            if line.startswith("<") or line.endswith(">"):
                continue
            if "最终答案" in line or "内部" in line or "思考过程" in line:
                continue
            lines.append(line)
        lines = lines[:6]
        return lines if len(lines) >= 3 else self.rule_based_case_analysis(question, evidence, answer)

    def rule_based_case_analysis(self, question: str, evidence: list[dict], answer: str) -> list[str]:
        question_text = str(question or "").strip()
        answer_text = str(answer or "").strip().replace("\n", " ")
        top_sources = [
            str(row.get("title") or row.get("source_label") or row.get("collection") or "检索来源").strip()
            for row in evidence[:4]
        ]
        source_text = "；".join(top_sources) if top_sources else "当前检索结果"
        issue_hint = "离婚财产、共同财产处分和是否存在隐藏、转移、变卖、毁损、挥霍夫妻共同财产等事实" if any(word in question_text for word in ["离婚", "夫妻", "婚姻", "共同财产"]) else "用户主张能否落到具体法律关系、关键事实和可证明的损失或请求"
        evidence_hint = "、".join(top_sources[:2]) if top_sources else "已上传或可补充的原始材料"
        conclusion_hint = answer_text[:180] if answer_text else "正式答案中的结论仍需要结合证据核验"
        return [
            f"实际争点：这次问题不能只看抽象法条，真正要先确认的是{issue_hint}；这些事实能不能被证据证明，会直接影响法院是否支持你的诉求。",
            f"现有证据怎么看：本次检索到的{source_text}有参考价值，但它们主要解决规则和裁判思路问题；如果缺少合同、转账、聊天记录、判决书等原始材料，仍然难以把规则稳定套到你的具体情况。",
            f"法律上怎么判断：需要把你的问题“{question_text[:80]}”拆成法律关系、行为经过、损害后果和请求目标，再用检索到的法条、解释或类案判断责任是否成立，而不是只凭来源标题下结论。",
            f"类似情况的处理倾向：如果{evidence_hint}能够证明关键事实，处理上通常会围绕举证责任、过错或违约程度、损失范围和程序选择来判断；如果证据链断裂，结论就会明显变弱。",
            f"接下来最实际的动作：先对照正式答案中的判断整理证据，尤其核对“{conclusion_hint}”这一结论需要哪些事实支撑；证据补齐后，再决定协商、发函、调解、投诉或起诉。",
        ]

    def answer_metadata(self, evidence: list[dict]) -> dict:
        citations = build_citations(evidence)
        public_titles = [c["title"] for c in citations if c["source_type"] == "public"]
        public_case_titles = [c["title"] for c in citations if c["source_type"] == "public" and c.get("collection") == "civil_cases"]
        private_titles = [c["title"] for c in citations if c["source_type"] in {"private", "user_upload"}]
        web_titles = [c["title"] for c in citations if c["source_type"] == "web"]
        evidence_analysis = [
            f'[{item.get("index", "?")}] {item.get("source_label", "未知来源")}：{item.get("title", "未命名来源")}'
            for item in citations[:6]
        ]
        if private_titles:
            evidence_analysis.append("用户材料：" + "、".join(private_titles[:4]))
        related_cases = [f"相关案例：{title}" for title in public_case_titles[:4]]
        if not related_cases:
            related_cases = [f"参考网页资料：{title}" for title in web_titles[:2]]
        return {
            "key_issues": [],
            "evidence_analysis": evidence_analysis,
            "legal_basis": [f"参考公共法律知识库：{title}" for title in public_titles[:4]],
            "related_cases": related_cases,
            "action_steps": ["核对回答中引用的来源。", "补充关键证据后继续追问。", "重大权益事项建议咨询执业律师。"],
            "document_checklist": ["合同或协议", "付款凭证", "沟通记录", "身份或主体证明"],
            "questions_to_confirm": ["是否还有未上传的关键材料？", "对方是否已经提出抗辩或拒绝理由？"],
            "suggestions": ["优先围绕证据和请求权基础补充事实。"],
            "citations": citations,
            "risk_notice": "本回答基于当前检索到的资料生成，仅供法律信息参考；重大权益事项建议咨询执业律师。",
        }

    @staticmethod
    def solution_title(question: str) -> str:
        text = str(question or "").strip()
        if any(word in text for word in ("殴打", "打人", "受伤", "伤情", "正当防卫", "人身损害")):
            return "人身损害（被殴打）维权指南"
        if any(word in text for word in ("离婚", "夫妻", "婚姻", "抚养", "彩礼")):
            return "婚姻家庭纠纷处理指南"
        if any(word in text for word in ("工资", "劳动合同", "辞退", "裁员", "工伤", "加班")):
            return "劳动争议维权指南"
        if any(word in text for word in ("借款", "欠款", "不还", "借条", "还款")):
            return "借款纠纷追偿指南"
        if any(word in text for word in ("合同", "违约", "退款", "定金", "解除")):
            return "合同纠纷处理指南"
        return "法律问题处理指南"

    def stream_answer_text(self, question: str, evidence: list[dict], history: list[dict] | None = None, thinking_enabled: bool | None = None):
        context = self._evidence_context(evidence, max_chars=model_setting_value(self.model, "answer_context_max_chars", 800))
        has_private_material = any(row.get("source_type") in {"private", "user_upload"} for row in evidence)
        try:
            yield from self._call_model(
                self.model.stream_chat,
                build_answer_messages(question, context, history, has_private_material, use_compact=True),
                thinking_enabled,
                model_setting_value(self.model, "answer_max_tokens", 2200),
                model_setting_float(self.model, "answer_llm_temperature", 0.25),
            )
        except Exception:
            logger.warning("LLM 流式生成失败，准备降级到非流式或本地兜底答案", extra={"event": "llm_stream_fallback"}, exc_info=True)
            try:
                fallback = self.generate(question, evidence, history, thinking_enabled=False).get("answer", "")
            except Exception:
                citations = build_citations(evidence)
                fallback = self._fallback_answer(question, evidence, citations) if evidence else self._no_evidence_answer(question)
            for char in str(fallback or ""):
                yield char

    @staticmethod
    def _no_evidence_answer(question: str) -> str:
        text = str(question or "").strip()
        if any(word in text for word in ["打", "动手", "反击", "正当防卫", "互殴", "争执", "伤"]):
            return (
                "你现在最关键的不是先给自己下结论，而是把“谁先动手、冲突是否还在持续、反击有没有超过必要限度”这条事实线固定住。只有这条线清楚了，后面才好判断是正当防卫、互殴，还是需要承担相应责任。\n\n"
                "---\n\n"
                "## 一、目前先看什么\n\n"
                "如果能证明对方先实施侵害，而且你是在侵害仍在进行时为了脱离危险或制止伤害而反击，**方向上可以主张正当防卫**。但如果对方已经停止，你继续追打，或者反击明显超过必要限度，就可能被认定为互殴、防卫过当，甚至产生治安、民事或刑事风险。\n\n"
                "## 二、还需要固定什么\n\n"
                "现在优先保存监控、现场视频、目击者信息、报警记录、聊天记录、病历、诊断证明、票据和伤情照片。做笔录时要把关键事实说完整：**对方怎样先动手、你为什么必须反击、反击持续了多久、有没有及时停止**。\n\n"
                "## 三、下一步怎么做\n\n"
                "已经报警的，后续可以申请调取监控、补充笔录、申请伤情鉴定；如果处理结果对你不利，再结合处罚决定或调解记录判断是否复议、诉讼或另行索赔。"
            )
        if any(word in text for word in ("工资", "劳动", "老板", "公司", "离职", "辞退", "仲裁", "社保", "加班")):
            return (
                f"关于“{text}”，你现在先不要只停留在“对方欠我钱”这个结论上，关键是把劳动关系、工资标准、欠付期间和催要过程固定成证据。只要这些能对应起来，后面才好判断投诉、劳动仲裁或诉讼怎么走。\n\n"
                "---\n\n"
                "## 一、目前能先判断什么\n\n"
                "如果你确实为对方提供劳动、接受其管理，并且双方约定或实际形成了工资支付关系，**欠发工资通常可以作为劳动争议主张处理**。但当前没有可引用的检索来源和上传材料，所以还不能直接确认欠薪金额、期间、仲裁时效或具体补偿项目。\n\n"
                "## 二、你现有证据要补成什么样\n\n"
                "优先整理劳动合同、入职记录、考勤排班、工资条、银行流水、聊天记录、工作群记录、工牌、社保记录、离职或辞退通知。证据要能说明三件事：**你在给谁工作、工资怎么算、哪几个月没发或少发**。\n\n"
                "## 三、下一步建议怎么做\n\n"
                "先做一张时间线和欠薪明细表，再向公司书面催要并保留送达记录；仍不处理的，可以考虑向劳动监察投诉或申请劳动仲裁。正式提交前，建议核对当地仲裁委对材料格式、主体信息和时效的要求。"
            )
        if any(word in text for word in ("离婚", "夫妻", "出轨", "共同财产", "抚养", "彩礼", "房产")):
            return (
                f"关于“{text}”，现在最重要的是把感情破裂、财产范围和证据来源分开看。离婚能不能准、财产怎么分、过错或损害赔偿能不能主张，通常都要分别建立事实和证据。\n\n"
                "---\n\n"
                "## 一、目前能先判断什么\n\n"
                "如果双方确已无法共同生活，可以考虑协议离婚或诉讼离婚；涉及出轨、隐匿转移财产、子女抚养或大额财产时，**不要只凭单方怀疑下结论**，要先固定可核验的材料。\n\n"
                "## 二、财产和过错证据怎么整理\n\n"
                "财产部分先列清房产、车辆、存款、投资、债务、保险、经营收入等清单，并标注取得时间、登记人、出资来源和现有凭证。过错部分要保存聊天记录、照片视频、住宿或消费记录、证人线索等，但要注意取证合法性和完整上下文。\n\n"
                "## 三、下一步建议怎么做\n\n"
                "先完成资产清单和证据目录；担心对方转移财产的，及时保存线索，并咨询是否申请调查取证或财产保全。房产、股权、大额债务和子女抚养问题建议结合原件让当地律师审核。"
            )
        return (
            f"关于“{text}”，当前没有拿到可引用的本地法条、案例或用户材料，所以只能先按法律问题的处理框架做初步判断。重点不是先套一个结论，而是把双方关系、争议事实、证据和请求事项拆清楚。\n\n"
            "---\n\n"
            "## 一、目前先判断什么\n\n"
            "先确认双方是什么法律关系，例如合同、借款、侵权、婚姻家庭、劳动争议或其他关系；再看争议发生的时间、金额、履行情况、对方态度和你想主张的结果。**法律路径不同，证据和程序也会不同**。\n\n"
            "## 二、现在最需要补什么\n\n"
            "优先整理合同或协议、付款凭证、聊天记录、通知函、现场照片、录音录像、证人信息、病历票据或已有文书。证据要尽量直接证明“发生了什么、谁做了什么、造成什么后果、你要求什么”。\n\n"
            "## 三、下一步建议怎么做\n\n"
            "在事实未固定前，不建议直接下确定结论。可以先协商或发函固定立场；对方拒不处理的，再结合证据选择投诉、调解、仲裁或起诉。"
        )

    @staticmethod
    def _no_evidence_metadata(question: str) -> dict:
        text = str(question or "")
        common = {
            "key_issues": ["当前没有可引用来源，只能先按用户陈述作一般法律方向判断。"],
            "evidence_analysis": ["未检索到可引用的本地知识库、网页资料或用户上传材料；需要补充能对应关键事实的证据。"],
            "legal_basis": [],
            "related_cases": [],
            "citations": [],
        }
        if any(word in text for word in ("工资", "劳动", "加班", "辞退", "离职", "工伤", "社保", "劳动合同")):
            return {
                **common,
                "action_steps": ["整理劳动关系、入离职、工资标准和欠付金额的时间线。", "保存劳动合同、工资流水、考勤、排班、聊天记录和催款记录。", "根据证据情况选择协商、投诉或劳动仲裁。"],
                "document_checklist": ["劳动合同或入职证明", "工资条、银行流水或转账记录", "考勤、排班、加班通知", "离职、辞退、催款等沟通记录"],
                "questions_to_confirm": ["是否有劳动合同或实际用工证明？", "欠付工资或补偿对应的期间和金额是多少？", "是否已经离职、被辞退或申请过仲裁？"],
                "suggestions": ["先把劳动关系和金额证据固定，再判断投诉、仲裁或诉讼路径。"],
                "risk_notice": "当前回答没有引用到检索来源，只能作为一般劳动争议信息参考；仲裁时效和金额计算需结合现行依据核验。",
            }
        if any(word in text for word in ("离婚", "夫妻", "婚姻", "共同财产", "抚养", "彩礼")):
            return {
                **common,
                "action_steps": ["列明婚姻登记、子女、共同财产、共同债务和争议焦点。", "保存房产、车辆、存款、投资、转账、聊天记录等原始材料。", "如担心财产转移，及时固定线索并咨询是否申请调查取证或保全。"],
                "document_checklist": ["结婚证、身份证明和户籍材料", "房产、车辆、银行流水、投资或债务凭证", "子女抚养相关支出和照护证据", "对方隐匿、转移或处分财产的线索"],
                "questions_to_confirm": ["双方是否已经登记结婚、是否有子女？", "争议财产取得时间和出资来源是什么？", "是否存在隐匿、转移、变卖或挥霍共同财产的证据？"],
                "suggestions": ["婚姻财产问题先做资产清单和证据目录，不要凭猜测下结论。"],
                "risk_notice": "当前回答没有引用到检索来源，只能作为一般婚姻家庭法律信息参考；房产、子女抚养和大额财产建议结合原件咨询律师。",
            }
        if any(word in text for word in ("打", "动手", "反击", "正当防卫", "互殴", "争执", "伤")):
            return {
                **common,
                "action_steps": ["先固定现场证据和沟通记录。", "及时报警、就医或申请调取监控。", "后续根据处理结果考虑复议、诉讼或索赔。"],
                "document_checklist": ["报警记录或接处警回执", "监控录像或现场视频", "证人联系方式", "病历、诊断证明和费用票据", "聊天记录或其他沟通证据"],
                "questions_to_confirm": ["对方先动手的过程有没有监控或证人？", "你反击时对方侵害是否仍在持续？", "双方是否有人受伤并做过伤情鉴定？"],
                "suggestions": ["先把事实经过和证据固定下来，再判断是否构成正当防卫或防卫过当。"],
                "risk_notice": "当前回答没有引用到检索来源，只能作为一般法律信息参考；涉及处罚、伤情鉴定或刑事风险时建议尽快咨询律师。",
            }
        return {
            **common,
            "action_steps": ["把争议事实按时间线整理清楚。", "保存合同、付款、沟通、通知和损失凭证等原始材料。", "根据证据情况选择协商、书面催告、调解、仲裁或起诉。"],
            "document_checklist": ["合同、协议或交易凭证", "付款、转账、收据或发票", "聊天记录、通知函或平台记录", "能证明损失或履行情况的材料"],
            "questions_to_confirm": ["双方是什么法律关系？", "争议发生的时间、金额或标的是什么？", "对方是否已经明确拒绝或提出抗辩？"],
            "suggestions": ["先补齐事实和证据，再判断具体请求和程序。"],
            "risk_notice": "当前回答没有引用到检索来源，只能作为一般法律信息参考；重大权益事项建议咨询执业律师。",
        }

    def generate(self, question: str, evidence: list[dict], history: list[dict] | None = None, thinking_enabled: bool | None = None) -> dict:
        citations = build_citations(evidence)
        context = self._evidence_context(evidence, max_chars=model_setting_value(self.model, "answer_context_max_chars", 800)) if evidence else "当前未提供有效检索证据。"
        has_private_material = any(row.get("source_type") in {"private", "user_upload"} for row in evidence)
        if not evidence:
            try:
                raw_answer = str(self._call_model(
                    self.model.chat,
                    build_no_evidence_answer_messages(question, history, has_private_material),
                    thinking_enabled,
                    model_setting_value(self.model, "no_evidence_answer_max_tokens", 1600),
                    model_setting_float(self.model, "no_evidence_llm_temperature", 0.2),
                ) or "").strip()
                answer = self._clean_model_output(raw_answer, citations)
            except Exception:
                logger.warning("无证据时 LLM 生成失败，使用本地兜底答案", extra={"event": "llm_no_evidence_fallback"}, exc_info=True)
                answer = ""
            if not answer:
                answer = self._no_evidence_answer(question)
            metadata = self._no_evidence_metadata(question)
            return {
                "answer": answer,
                "case_analysis": self.build_case_analysis(question, evidence, answer, thinking_enabled=False),
                **metadata,
            }
        try:
            raw_answer = str(self._call_model(
                self.model.chat,
                build_answer_messages(question, context, history, has_private_material, use_compact=True),
                thinking_enabled,
                model_setting_value(self.model, "answer_max_tokens", 2200),
                model_setting_float(self.model, "answer_llm_temperature", 0.25),
            ) or "").strip()
            answer = self._clean_model_output(raw_answer, citations)
        except Exception:
            logger.warning("LLM 非流式生成失败，使用本地兜底答案", extra={"event": "llm_sync_fallback"}, exc_info=True)
            answer = self._fallback_answer(question, evidence, citations)
        if not answer:
            answer = self._fallback_answer(question, evidence, citations)
        answer = self.ensure_source_highlights(answer, question, evidence)
        metadata = self.answer_metadata(evidence)
        return {"answer": answer, "case_analysis": self.build_case_analysis(question, evidence, answer, thinking_enabled=False), **metadata}

    @staticmethod
    def _fallback_solution(question: str, answer: str, evidence: list[dict]) -> str:
        question_text = str(question or "本次法律咨询").strip() or "本次法律咨询"
        answer_text = str(answer or "").strip()
        source_titles = []
        for row in evidence[:6]:
            title = str(row.get("title") or row.get("source_label") or "检索来源").strip()
            if title and title not in source_titles:
                source_titles.append(title)
        source_text = "\n".join(f"- {title}" for title in source_titles) or "- 当前没有可稳定列出的参考来源"
        status = "🟢 方向较明确" if evidence else "🟠 需要重点补证"
        core = answer_text[:360] if answer_text else "当前信息还不足，需要先补齐事实经过和关键证据，再判断具体处理路径。"
        title = AnswerGenerator.solution_title(question_text)
        return f"""# {title}
*基于现有信息的初步方案*

## 1. 先给用户的结论
{core}

1. **先固定事实和证据**，保留原始文件、时间和完整上下文。
2. **核对对方身份、责任和处理期限**，不要先作不可逆的承诺或处置。
3. **根据证据选择协商、投诉、调解、仲裁或诉讼路径**。

## 2. 立即行动清单（优先级最高）
### 2.1 当前最先做的事
**先把能证明关键事实的材料整理成时间线，并为每项材料编号。**

### 2.2 证据清单
- 事实发生经过、时间、地点和相关人员记录；
- 合同、通知、付款凭证或其他权利义务文件；
- 聊天记录、录音录像、照片和平台记录，保留原始载体；
- 能证明损失、履行情况或对方行为后果的凭证；
- 对方身份、联系方式和已提出的理由或抗辩。

### 2.3 取证必须合法
不得盗取账号、非法监听、侵入住宅、威胁取证、伪造或剪辑证据。电子数据应保留来源、形成过程和完整上下文。

## 3. 识别责任路径
| 事实情况 | 可能路径 | 识别重点 |
| --- | --- | --- |
| {question_text[:80]} | 依据现有事实和证据初步判断 | 还需核对关键行为、责任主体和结果 |
| 证据尚不完整 | 补充材料后再确定救济路径 | 时间线、原始文件和损失凭证是否齐全 |
| 对方拒绝处理或存在期限 | 协商、投诉、调解、仲裁或诉讼 | 管辖、时效和程序材料不能拖延 |

## 4. 可主张的责任、请求或赔偿
- 要求对方继续履行、返还、赔偿损失或停止侵害；
- 根据实际后果主张合理费用和可证明的损失；
- 符合法定条件时，主张违约责任、侵权责任或其他救济。

具体金额、责任范围和处理结果，需要结合事实、证据、过错程度和实际损失计算，不能凭主观估算。

## 5. 起诉/申请/维权前的四项整理
### 5.1 原告/申请人信息
准备身份证明、联系方式、住所或经常居住地等基础信息。

### 5.2 对方/被告信息
尽量核实姓名、主体名称、联系方式和可送达地址；不完整时通过合法程序补齐。

### 5.3 事实时间线
按“日期 → 地点 → 行为 → 后果 → 沟通/处理 → 损失”记录，并附对应证据编号。

### 5.4 请求与证据目录
将每一项请求与合同、记录、票据、鉴定或其他证据逐项关联。

## 6. 管辖、立案与时效
根据争议类型核对处理机关或法院、管辖连接点、立案材料和诉讼时效。正式行动前应核对当地现行要求，发现可能临近期限时及时咨询专业人士。

## 7. 明确禁止的报复性维权
- 不要公开对方身份证、住址、手机号等个人信息；
- 不要威胁、恐吓、围攻、私自扣押或骚扰对方；
- 不要未经核实公开指控对方犯罪；
- 不要删除、剪辑或伪造证据。

## 8. 面向用户的简明输出模板
**初步判断：** {status}。当前结论仍需结合完整事实和证据核验。

**现在先做：** 先整理时间线和原始证据，保留提交、报警、沟通或就医记录。

**如果仍有危险/紧急情形：** 先离开危险现场，联系当地紧急服务，不要单独与对方见面。

**后续可主张：** 根据责任路径和可证明损失，主张返还、履行、赔偿、停止侵害或其他法定救济。

**需要补充：** 对方是谁、何时何地发生了什么、已有哪类证据、造成什么损失，以及是否存在处理期限。

## 9. 法律依据索引（用于检索标注）
{source_text}

法律和司法解释可能修订。正式报案、起诉、申请或计算损失前，应核对现行有效文本及当地主管机关要求。
""".strip()

    def generate_solution(self, question: str, answer: str | dict, evidence: list[dict], history: list[dict] | None = None, thinking_enabled: bool | None = None) -> dict:
        answer_text = str(answer.get("answer", "") if isinstance(answer, dict) else answer or "").strip()
        context = self._evidence_context(evidence, max_chars=model_setting_value(self.model, "answer_context_max_chars", 800))
        has_private_material = any(row.get("source_type") in {"private", "user_upload"} for row in evidence)
        start = perf_counter()
        try:
            raw_markdown = str(self._call_model(
                self.model.chat,
                build_solution_messages(question, answer_text, context, history, has_private_material),
                thinking_enabled=False,
                max_tokens=model_setting_value(self.model, "solution_max_tokens", 1800),
                temperature=model_setting_float(self.model, "solution_llm_temperature", 0.35),
            ) or "").strip()
            markdown = self._clean_model_output(raw_markdown, build_citations(evidence))
            if markdown.startswith("```"):
                lines = markdown.splitlines()
                if lines and lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                markdown = "\n".join(lines).strip()
            heading_match = re.search(r"(?m)^#\s+\S", markdown)
            if heading_match and heading_match.start() > 0:
                markdown = markdown[heading_match.start():].strip()
            markdown = self._clean_model_output(markdown, build_citations(evidence))
        except Exception:
            logger.warning("LLM 法律解决方案生成失败，使用本地兜底方案", extra={"event": "llm_solution_fallback"}, exc_info=True)
            markdown = ""
        if not markdown:
            markdown = self._fallback_solution(question, answer_text, evidence)
        if not re.match(r"^#\s+\S", markdown):
            markdown = self._fallback_solution(question, answer_text, evidence)
        elapsed_ms = round((perf_counter() - start) * 1000, 2)
        return {
            "title": self.solution_title(question),
            "scope": "第二次模型调用生成",
            "markdown": markdown,
            "elapsed_ms": elapsed_ms,
            "citations": build_citations(evidence),
        }


PAGE_WIDTH = 1240
PAGE_HEIGHT = 1754
MARGIN_X = 105
MARGIN_Y = 100
CONTENT_WIDTH = PAGE_WIDTH - (MARGIN_X * 2)
BODY_SIZE = 30
HEADING_SIZE = 42
SUBHEADING_SIZE = 34
LINE_GAP = 18
TEXT_FILL = (35, 35, 35)
MUTED_FILL = (98, 100, 93)
TABLE_BORDER = (216, 224, 230)
TABLE_HEADER_BG = (217, 234, 247)
TABLE_HEADER_FILL = (41, 72, 92)
PRIORITY_BG = (255, 249, 233)
PRIORITY_ACCENT = (197, 164, 95)
TABLE_CELL_PADDING_X = 14
TABLE_CELL_PADDING_Y = 12


def _font_candidates(bold: bool = False) -> list[Path]:
    names = (
        ["msyhbd.ttc", "simhei.ttf", "NotoSansCJK-Bold.ttc", "NotoSerifCJK-Bold.ttc"]
        if bold
        else ["msyh.ttc", "simsun.ttc", "NotoSansCJK-Regular.ttc", "NotoSerifCJK-Regular.ttc"]
    )
    roots = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts/opentype/noto"),
        Path("/usr/share/fonts/truetype/noto"),
        Path("/usr/share/fonts/truetype/dejavu"),
    ]
    return [root / name for root in roots for name in names]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _font_candidates(bold):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _wrap_line(text: str, font: ImageFont.ImageFont, width: int = CONTENT_WIDTH) -> list[str]:
    if not text:
        return [""]
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and draw.textlength(candidate, font=font) > width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _strip_inline_markdown(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _split_table_cells(line: str) -> list[str]:
    value = str(line or "").strip()
    if not value.startswith("|") or "|" not in value[1:]:
        return []
    return [cell.strip() for cell in value.strip("|").split("|")]


def _is_table_divider(line: str) -> bool:
    cells = _split_table_cells(line)
    return len(cells) >= 2 and all(re.match(r"^:?-{3,}:?$", cell) for cell in cells)


def _table_column_widths(column_count: int) -> list[int]:
    if column_count <= 0:
        return []
    if column_count == 3:
        ratios = [0.31, 0.34, 0.35]
    else:
        ratios = [1 / column_count] * column_count
    widths = [int(CONTENT_WIDTH * ratio) for ratio in ratios]
    widths[-1] += CONTENT_WIDTH - sum(widths)
    return widths


def _content_blocks(markdown: str) -> list[dict]:
    blocks: list[dict] = []
    body_font = _load_font(BODY_SIZE)
    body_bold_font = _load_font(BODY_SIZE, bold=True)
    heading_font = _load_font(HEADING_SIZE, bold=True)
    subheading_font = _load_font(SUBHEADING_SIZE, bold=True)
    raw_lines = str(markdown or "").replace("\r\n", "\n").split("\n")
    index = 0
    while index < len(raw_lines):
        raw_line = raw_lines[index]
        line = raw_line.strip()
        if not line:
            blocks.append({"type": "text", "text": "", "font": body_font, "top_gap": 4, "bottom_gap": 4, "fill": TEXT_FILL})
            index += 1
            continue
        if line.startswith("# "):
            blocks.append({"type": "heading", "text": _strip_inline_markdown(line[2:]), "font": heading_font, "top_gap": 26, "bottom_gap": 18})
            index += 1
            continue
        elif line.startswith("## "):
            text, font, top_gap, bottom_gap = _strip_inline_markdown(line[3:]), subheading_font, 22, 12
        elif line.startswith("### "):
            text, font, top_gap, bottom_gap = _strip_inline_markdown(line[4:]), body_bold_font, 18, 10
        elif line.startswith("#### "):
            text, font, top_gap, bottom_gap = _strip_inline_markdown(line[5:]), body_bold_font, 16, 8
        elif _split_table_cells(line) and index + 1 < len(raw_lines) and _is_table_divider(raw_lines[index + 1]):
            header_cells = [_strip_inline_markdown(cell) for cell in _split_table_cells(line)]
            rows = []
            row_index = index + 2
            while row_index < len(raw_lines):
                row_cells = _split_table_cells(raw_lines[row_index])
                if len(row_cells) != len(header_cells) or not row_cells:
                    break
                rows.append([_strip_inline_markdown(cell) for cell in row_cells])
                row_index += 1
            blocks.append({"type": "table", "header": header_cells, "rows": rows, "top_gap": 12, "bottom_gap": 18})
            index = row_index
            continue
        elif line.startswith(("- ", "* ")):
            text, font, top_gap, bottom_gap = f"• {_strip_inline_markdown(line[2:])}", body_font, 4, 4
        elif re.match(r"^\d+[.)]\s+", line):
            text, font, top_gap, bottom_gap = _strip_inline_markdown(line), body_font, 4, 4
        elif line.startswith(">"):
            text, font, top_gap, bottom_gap = _strip_inline_markdown(line.lstrip("> ")), body_font, 8, 4
        else:
            text, font, top_gap, bottom_gap = _strip_inline_markdown(line), body_font, 4, 4
        blocks.append({"type": "text", "text": text, "font": font, "top_gap": top_gap, "bottom_gap": bottom_gap, "fill": MUTED_FILL if line.startswith(">") else TEXT_FILL})
        index += 1
    return blocks


def _new_pdf_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    return page, ImageDraw.Draw(page)


def _line_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text or " ", font=font)
    return max(42, bbox[3] - bbox[1] + LINE_GAP)


def _text_block_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int = CONTENT_WIDTH) -> tuple[list[str], int]:
    lines = _wrap_line(text, font, width)
    return lines, sum(_line_height(draw, line, font) for line in lines)


def _draw_text_block(draw: ImageDraw.ImageDraw, y: int, block: dict) -> int:
    text = str(block.get("text") or "")
    font = block["font"]
    lines, _height = _text_block_height(draw, text, font)
    for line in lines:
        draw.text((MARGIN_X, y), line, fill=block.get("fill", TEXT_FILL), font=font)
        y += _line_height(draw, line, font)
    return y


def _draw_heading_block(draw: ImageDraw.ImageDraw, y: int, block: dict) -> int:
    text = str(block.get("text") or "")
    font = block["font"]
    lines, text_height = _text_block_height(draw, text, font, CONTENT_WIDTH - 42)
    block_height = text_height + 32
    draw.rounded_rectangle((MARGIN_X, y, PAGE_WIDTH - MARGIN_X, y + block_height), radius=4, fill=PRIORITY_BG)
    draw.rectangle((MARGIN_X, y, MARGIN_X + 8, y + block_height), fill=PRIORITY_ACCENT)
    text_y = y + 16
    for line in lines:
        draw.text((MARGIN_X + 24, text_y), line, fill=TEXT_FILL, font=font)
        text_y += _line_height(draw, line, font)
    return y + block_height


def _table_row_layout(draw: ImageDraw.ImageDraw, cells: list[str], widths: list[int], font: ImageFont.ImageFont) -> tuple[list[list[str]], int]:
    wrapped_cells: list[list[str]] = []
    max_height = 0
    for cell, width in zip(cells, widths):
        lines = _wrap_line(_strip_inline_markdown(cell), font, max(24, width - (TABLE_CELL_PADDING_X * 2)))
        wrapped_cells.append(lines)
        max_height = max(max_height, sum(_line_height(draw, line, font) for line in lines))
    return wrapped_cells, max_height + (TABLE_CELL_PADDING_Y * 2)


def _draw_table_row(
    draw: ImageDraw.ImageDraw,
    y: int,
    cells: list[str],
    widths: list[int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] | None = None,
    text_fill: tuple[int, int, int] = TEXT_FILL,
) -> int:
    wrapped_cells, row_height = _table_row_layout(draw, cells, widths, font)
    x = MARGIN_X
    for lines, width in zip(wrapped_cells, widths):
        if fill:
            draw.rectangle((x, y, x + width, y + row_height), fill=fill)
        draw.rectangle((x, y, x + width, y + row_height), outline=TABLE_BORDER, width=1)
        text_y = y + TABLE_CELL_PADDING_Y
        for line in lines:
            draw.text((x + TABLE_CELL_PADDING_X, text_y), line, fill=text_fill, font=font)
            text_y += _line_height(draw, line, font)
        x += width
    draw.line((MARGIN_X, y, MARGIN_X + sum(widths), y), fill=TABLE_BORDER, width=1)
    draw.line((MARGIN_X, y + row_height, MARGIN_X + sum(widths), y + row_height), fill=TABLE_BORDER, width=1)
    return y + row_height


def render_solution_pdf(markdown: str) -> bytes:
    pages: list[Image.Image] = []
    page, draw = _new_pdf_page()
    y = MARGIN_Y

    def add_page() -> None:
        nonlocal page, draw, y
        pages.append(page)
        page, draw = _new_pdf_page()
        y = MARGIN_Y

    for block in _content_blocks(markdown):
        block_type = block.get("type")
        top_gap = int(block.get("top_gap", 0))
        bottom_gap = int(block.get("bottom_gap", 0))
        if block_type == "table":
            header = list(block.get("header") or [])
            rows = list(block.get("rows") or [])
            widths = _table_column_widths(len(header))
            if not header or not widths:
                continue
            header_font = _load_font(BODY_SIZE, bold=True)
            body_font = _load_font(BODY_SIZE)
            header_height = _table_row_layout(draw, header, widths, header_font)[1]
            y += top_gap
            if y + header_height > PAGE_HEIGHT - MARGIN_Y:
                add_page()
            y = _draw_table_row(draw, y, header, widths, header_font, fill=TABLE_HEADER_BG, text_fill=TABLE_HEADER_FILL)
            for row in rows:
                row_height = _table_row_layout(draw, row, widths, body_font)[1]
                if y + row_height > PAGE_HEIGHT - MARGIN_Y:
                    add_page()
                    y = _draw_table_row(draw, y, header, widths, header_font, fill=TABLE_HEADER_BG, text_fill=TABLE_HEADER_FILL)
                y = _draw_table_row(draw, y, row, widths, body_font)
            y += bottom_gap
            continue

        text = str(block.get("text") or "")
        font = block["font"]
        if block_type == "heading":
            lines, text_height = _text_block_height(draw, text, font, CONTENT_WIDTH - 42)
            block_height = text_height + 32
        else:
            lines, block_height = _text_block_height(draw, text, font)
        if y + top_gap + block_height > PAGE_HEIGHT - MARGIN_Y:
            add_page()
        y += top_gap
        if block_type == "heading":
            y = _draw_heading_block(draw, y, block)
        else:
            y = _draw_text_block(draw, y, block)
        y += bottom_gap
    if page not in pages:
        pages.append(page)
    if not pages:
        page, _draw = _new_pdf_page()
        pages.append(page)

    output = io.BytesIO()
    pages[0].save(output, format="PDF", resolution=150.0, save_all=True, append_images=pages[1:])
    return output.getvalue()
