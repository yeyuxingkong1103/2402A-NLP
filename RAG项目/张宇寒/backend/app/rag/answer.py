"""根据检索证据调用大语言模型生成回答、方案，并把方案渲染为 PDF。

这个文件是在线 RAG 的最后一层：把问题、对话历史和证据组织成提示词，调用大模型，
清理模型输出，补充引用与风险提示；模型不可用时提供本地兜底。后半部分负责将方案
Markdown 排版为可下载的多页 PDF。
"""

# io.BytesIO 在内存中生成 PDF 字节，不必先写临时文件。
import io
# logging 记录模型生成或方案降级。
import logging
# Path 搜索 Windows/Linux 中文字体文件。
from pathlib import Path
# re 清理模型标签、Markdown 和提取字段。
import re
# perf_counter 统计第二次模型调用生成方案的耗时。
from time import perf_counter

# Pillow 创建 PDF 页面、绘制文字/表格并加载字体。
from PIL import Image, ImageDraw, ImageFont

# 公共集合中文标签用于告诉用户依据来自法条、案例还是其他库。
from ..core import PUBLIC_COLLECTION_LABELS
# 模型网关和提示词/输出校验函数。
from ..models import ModelGateway, build_answer_messages, build_no_evidence_answer_messages, build_solution_messages, extract_final_answer, validate_final_answer
# 统一证据节选规则，法律规范保持全文，其他资料明确标注节选。
from .context import evidence_excerpt


# 独立日志器便于观察答案生成和降级。
logger = logging.getLogger("law_rag.answer")

# 非公共集合的用户可见来源名称。
SOURCE_LABELS = {
    "public": "公共法律库",
    "private": "用户上传材料",
    "user_upload": "用户上传材料",
    "web": "网页资料",
}


def model_setting_value(model: ModelGateway, name: str, default: int) -> int:
    """从模型配置读取正整数，缺失或非法时使用默认值。"""

    # settings 可能不存在于测试替身上，因此逐层 getattr。
    value = getattr(getattr(model, "settings", None), name, default)
    try:
        return max(1, int(value or default))
    except (TypeError, ValueError):
        return default


def model_setting_float(model: ModelGateway, name: str, default: float) -> float:
    """从模型配置读取浮点数，非法时使用默认值。"""

    value = getattr(getattr(model, "settings", None), name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def answer_token_budget(base_tokens: int, answer_detail: str) -> int:
    """把前端回答详细度转换为有上下限的模型输出 token 数。"""

    # 基础预算至少为 1。
    base = max(1, int(base_tokens or 1))
    # 简洁回答最多 900 token。
    if answer_detail == "concise":
        return min(base, 900)
    # 详细回答在基础上增加约 30%，但最多 3600。
    if answer_detail == "detailed":
        return min(3600, max(base + 1, int(base * 1.3)))
    # 标准回答使用配置基础值。
    return base


def build_citations(rows: list[dict]) -> list[dict]:
    """把内部证据记录转换成前端可展示的编号引用列表。"""

    # enumerate 从 1 编号，使正文可以使用 [1]、[2] 引用。
    return [
        {
            "index": i,
            "title": row.get("title", "未知来源"),
            "source_type": row.get("source_type", "unknown"),
            # 公共来源优先显示具体集合中文名，其他来源用 SOURCE_LABELS。
            "source_label": row.get("source_label") or (PUBLIC_COLLECTION_LABELS.get(row.get("collection", ""), "公共法律库") if row.get("source_type") == "public" else SOURCE_LABELS.get(row.get("source_type", "unknown"), "未知来源")),
            "source_id": row.get("source_id", ""),
            "collection": row.get("collection", ""),
            "file_name": row.get("file_name", ""),
            # 网页结果若 source_id 本身是 URL，也作为可点击地址。
            "url": row.get("url") or row.get("link") or (row.get("source_id", "") if row.get("source_type") == "web" and str(row.get("source_id", "")).startswith(("http://", "https://")) else ""),
        }
        for i, row in enumerate(rows, start=1)
    ]


class AnswerGenerator:
    """生成普通回答、案件分析、详细方案和相关引用元数据。"""

    def __init__(self, model: ModelGateway):
        # 保存统一模型网关，流式/非流式/方案调用都复用它。
        self.model = model

    @staticmethod
    def _looks_like_signature_type_error(error: TypeError) -> bool:
        """判断 TypeError 是否仅由测试替身/旧模型函数参数签名不兼容引起。"""

        message = str(error)
        # 只有这些明确的参数错误才允许去掉可选参数重试。
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
        """兼容新旧模型函数签名调用，不能吞掉函数内部真正的 TypeError。"""

        try:
            # 有 token 预算时传完整新版参数。
            if max_tokens is not None:
                return func(messages, max_tokens=max_tokens, thinking_enabled=thinking_enabled, temperature=temperature)
            return func(messages, thinking_enabled=thinking_enabled, temperature=temperature)
        except TypeError as exc:
            # 如果不是“参数不支持”，说明是模型函数内部错误，原样抛出。
            if not AnswerGenerator._looks_like_signature_type_error(exc):
                raise
            try:
                # 第一次兼容重试只保留 max_tokens。
                if max_tokens is not None:
                    return func(messages, max_tokens=max_tokens)
                return func(messages)
            except TypeError as retry_exc:
                # 第二次仍是其他内部 TypeError 时也不能掩盖。
                if not AnswerGenerator._looks_like_signature_type_error(retry_exc):
                    raise
                # 最旧测试替身只接收 messages。
                return func(messages)

    @staticmethod
    def _clean_model_output(text: str, citations: list[dict] | None = None) -> str:
        """移除代码围栏和 final_answer 包装，并校验模型最终输出格式。"""

        value = str(text or "").strip()
        if not value:
            return ""
        # 模型偶尔会把整段答案包进 Markdown 代码块，先去掉首尾围栏。
        if value.startswith("```"):
            lines = value.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            value = "\n".join(lines).strip()
        # 校验 final_answer 标签和引用是否符合提示词协议。
        valid, issues = validate_final_answer(value, citations)
        if valid:
            # 只返回 final_answer 标签内部内容。
            return extract_final_answer(value)
        # 标签存在但不完整时尽力按边界修复，并写警告。
        if "<final_answer>" in value or "</final_answer>" in value:
            logger.warning(
                "模型输出 final_answer 标签不完整，按内容边界清洗",
                extra={"event": "final_answer_envelope_repaired", "fields": {"issues": issues[:4]}},
            )
            if "<final_answer>" in value:
                value = value.split("<final_answer>", 1)[1]
            if "</final_answer>" in value:
                value = value.split("</final_answer>", 1)[0]
        # 没有标签时保留模型正文，后续还有本地兜底和清理。
        return value.strip()

    @staticmethod
    def _source_details(row: dict) -> str:
        """从证据记录提取条号、文号、案号等关键字段，帮助模型准确引用。"""

        # details 按“标签=值”收集来源信息。
        details = []
        # 内部来源 ID 可用于日志核对和引用回溯。
        source_id = str(row.get("source_id") or "").strip()
        if source_id:
            details.append(f"来源ID={source_id}")
        # 优先直接读取 article_number。
        article_label = ""
        value = row.get("article_number")
        if value not in (None, ""):
            article_label = f"第{value}条"
        else:
            # 旧记录没有 article_number 时，尝试从稳定 source_id 中恢复条号。
            source_id_value = str(row.get("source_id") or "")
            collection = str(row.get("collection") or "")
            if collection == "civil_code_articles":
                match = re.search(r"article_(\d+)$", source_id_value)
                if match:
                    article_label = f"第{match.group(1)}条"
            elif collection == "civil_elements":
                match = re.search(r":(\d+)$", source_id_value)
                if match:
                    article_label = f"第{match.group(1)}条"
        # 找到条号后加入详细信息。
        if article_label:
            details.append(f"条号={article_label}")
        # 不同集合可能具有的业务字段及其中文标签。
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
        # 逐字段提取非空值。
        for field, label in field_labels:
            value = row.get(field)
            if value in (None, "", []):
                continue
            # 列表最多展示前五项，用顿号连接。
            if isinstance(value, list):
                value = "、".join(str(item) for item in value[:5])
            details.append(f"{label}={value}")
        # 分号分隔，作为证据正文前的“关键来源信息”。
        return "；".join(details)

    @staticmethod
    def _source_highlight_terms(row: dict) -> list[str]:
        """提取公共证据中回答应该明确提到的条号、标题、文号或争点。"""

        # 用户材料和网页没有统一公共字段，不做强制术语补充。
        if row.get("source_type") != "public" or not row.get("collection"):
            return []
        # 这些字段对用户核对法律依据最有帮助。
        fields = ("title", "document_number", "cause_of_action", "rule_name", "question")
        terms = []
        # 优先取得法条标签。
        article_label = ""
        value = row.get("article_number")
        if value not in (None, ""):
            article_label = f"第{value}条"
        else:
            source_id = str(row.get("source_id") or "")
            collection = str(row.get("collection") or "")
            # 兼容从旧主键恢复民法典或法律要素条号。
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
        # 逐字段把长度合理、尚未出现的文字加入术语列表。
        for field in fields:
            value = row.get(field)
            if value in (None, "", []):
                continue
            if isinstance(value, list):
                values = [str(item).strip() for item in value if str(item).strip()]
            else:
                values = [str(value).strip()]
            for item in values:
                # 太短缺少辨识度，太长不适合作为“必须提及术语”。
                if 2 <= len(item) <= 140 and item not in terms:
                    terms.append(item)
        # 案例再提取正文首行作为可能的裁判规则摘要。
        if row.get("collection") == "civil_cases":
            first_line = str(row.get("content") or "").strip().splitlines()[0:1]
            if first_line:
                snippet = first_line[0].strip()[:100]
                if len(snippet) >= 8 and snippet not in terms:
                    terms.append(snippet)
        return terms

    @staticmethod
    def ensure_source_highlights(answer: str, question: str, evidence: list[dict]) -> str:
        """若回答漏掉关键法条或来源名称，在末尾追加简短“依据核对”。"""

        # 没有答案或证据时不处理。
        value = str(answer or "").strip()
        if not value or not evidence:
            return value
        terms = []
        question_text = str(question or "")
        # 用户明确问构成要件等概念时，也要求回答中能够检索到这些词。
        for term in ("构成要件", "关键事实", "举证责任"):
            if term in question_text and term not in terms:
                terms.append(term)
        # 最多从前四条核心证据提取关键术语。
        for row in evidence[:4]:
            for term in AnswerGenerator._source_highlight_terms(row):
                if term not in terms:
                    terms.append(term)
        # 找出证据中存在、回答却完全未提到的术语。
        missing = [term for term in terms if term and term not in value]
        # 没有遗漏就保持原回答。
        if not missing:
            return value
        # 最多追加十项，提醒用户进一步核对，而不伪造新的法律分析。
        return f"{value}\n\n**依据核对**：{'；'.join(missing[:10])}。"

    @staticmethod
    def _evidence_context(evidence: list[dict], max_chars: int = 1200) -> str:
        """把结构化证据格式化为带编号、来源标签和关键字段的模型上下文。"""

        parts = []
        # 编号从 1 开始，与 citations 中的 [1]、[2] 保持一致。
        for i, row in enumerate(evidence, start=1):
            # 优先使用已有中文标签；私有材料和一般来源提供明确后备值。
            label = row.get("source_label") or ("用户上传材料" if row.get("source_type") in {"private", "user_upload"} else "检索来源")
            # 条号、文号、案号等作为关键来源信息。
            details = AnswerGenerator._source_details(row)
            detail_text = f"\n关键来源信息：{details}" if details else ""
            # 使用 ContextBuilder 相同裁剪规则，法条和司法解释保持完整。
            content = evidence_excerpt(row, max_chars)["content"]
            parts.append(f"[{i}] 【{label}】{row.get('title')}{detail_text}\n{content}")
        # 空行分隔来源，减少模型混淆相邻证据。
        return "\n\n".join(parts)

    def _fallback_answer(self, question: str, evidence: list[dict], citations: list[dict]) -> str:
        """有检索证据但模型生成失败时，按问题类型生成保守的本地回答。"""

        # 清理问题并识别是否主要询问证据、是否属于劳动场景。
        text = str(question or "").strip()
        evidence_question = any(marker in text for marker in ("证据", "材料", "证明", "举证"))
        labor_question = any(marker in text for marker in ("工资", "劳动", "加班", "辞退", "离职", "工伤"))
        # 明确来源只能帮助核对规则，不能代替用户证明案件事实。
        source_note = f"本轮检索到 {len(citations)} 条参考来源，但来源只用于核对规则，不等同于已经证明你的具体事实。"
        # 不同问题类型使用不同的行动清单和重点说明。
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
        # 拼装统一结构：共情说明、核心判断、行动、待确认和依据边界。
        return (
            f"我知道你现在最想尽快确认该准备什么、下一步怎么走。关于“{text}”，先把重点放在可证明的事实和证据上。\n\n"
            f"## 核心判断\n**{focus}**\n\n"
            f"## 现在先做\n{chr(10).join(f'{i}. {item}' for i, item in enumerate(actions, 1))}\n\n"
            f"## 还需确认\n核对双方身份关系、关键时间、金额或损失、对方当前态度，以及是否已经存在仲裁、诉讼或其他期限风险。\n\n"
            f"## 依据说明\n{source_note} 引用内容应结合原始材料进一步核验。\n\n"
            "以上内容仅供法律信息参考；涉及较大金额、劳动仲裁、诉讼期限或重大权益，建议尽快让执业律师审核原始材料。"
        )

    def build_case_analysis(self, question: str, evidence: list[dict], answer: str, thinking_enabled: bool | None = None) -> list[str]:
        """生成前端单独展示的案件分析要点；没有证据时强调缺口而不下结论。"""

        # 无证据时返回固定四步框架，明确事实和材料不足。
        if not evidence:
            return [
                "实际争点：现在还缺少能直接对应到事实和法律规则的材料，暂时不能把结论说死。",
                "现有证据怎么看：当前没有检索到足够依据，最关键的是先补齐事实经过、时间节点、金额或标的、双方关系和已有凭证。",
                "法律上怎么判断：需要先确定属于哪类法律关系，再看请求权基础、举证责任和诉讼时效；没有证据时不能只凭主观描述作确定判断。",
                "接下来最实际的动作：把合同、聊天记录、付款凭证、通知函、法院文书或其他能证明关键事实的材料补充后，再继续追问具体处理路径。",
            ]
        # 有证据时使用可解释的规则化分析，不再增加一次模型调用。
        return self.rule_based_case_analysis(question, evidence, answer)

    def rule_based_case_analysis(self, question: str, evidence: list[dict], answer: str) -> list[str]:
        """根据问题、证据标题和正式答案组织五条案件分析。"""

        # 清理问题和回答，用于生成简短提示。
        question_text = str(question or "").strip()
        answer_text = str(answer or "").strip().replace("\n", " ")
        # 最多取前四条证据的标题或来源名。
        top_sources = [
            str(row.get("title") or row.get("source_label") or row.get("collection") or "检索来源").strip()
            for row in evidence[:4]
        ]
        # 分号连接，作为现有证据说明。
        source_text = "；".join(top_sources) if top_sources else "当前检索结果"
        # 婚姻家庭问题使用更具体的争点提示，其他问题使用通用民事分析框架。
        issue_hint = "离婚财产、共同财产处分和是否存在隐藏、转移、变卖、毁损、挥霍夫妻共同财产等事实" if any(word in question_text for word in ["离婚", "夫妻", "婚姻", "共同财产"]) else "用户主张能否落到具体法律关系、关键事实和可证明的损失或请求"
        # 前两条来源用于说明还需对应哪些原始材料。
        evidence_hint = "、".join(top_sources[:2]) if top_sources else "已上传或可补充的原始材料"
        # 截取正式答案开头，提醒用户核对该结论需要哪些事实支持。
        conclusion_hint = answer_text[:180] if answer_text else "正式答案中的结论仍需要结合证据核验"
        # 五条分别覆盖争点、证据、法律判断、处理倾向和下一步行动。
        return [
            f"实际争点：这次问题不能只看抽象法条，真正要先确认的是{issue_hint}；这些事实能不能被证据证明，会直接影响法院是否支持你的诉求。",
            f"现有证据怎么看：本次检索到的{source_text}有参考价值，但它们主要解决规则和裁判思路问题；如果缺少合同、转账、聊天记录、判决书等原始材料，仍然难以把规则稳定套到你的具体情况。",
            f"法律上怎么判断：需要把你的问题“{question_text[:80]}”拆成法律关系、行为经过、损害后果和请求目标，再用检索到的法条、解释或类案判断责任是否成立，而不是只凭来源标题下结论。",
            f"类似情况的处理倾向：如果{evidence_hint}能够证明关键事实，处理上通常会围绕举证责任、过错或违约程度、损失范围和程序选择来判断；如果证据链断裂，结论就会明显变弱。",
            f"接下来最实际的动作：先对照正式答案中的判断整理证据，尤其核对“{conclusion_hint}”这一结论需要哪些事实支撑；证据补齐后，再决定协商、发函、调解、投诉或起诉。",
        ]

    def answer_metadata(self, evidence: list[dict]) -> dict:
        """从最终证据生成引用、依据、类案、行动和风险提示等结构化字段。"""

        # 转成前端编号引用。
        citations = build_citations(evidence)
        # 分来源筛选标题，供不同 UI 区域展示。
        public_titles = [c["title"] for c in citations if c["source_type"] == "public"]
        public_case_titles = [c["title"] for c in citations if c["source_type"] == "public" and c.get("collection") == "civil_cases"]
        private_titles = [c["title"] for c in citations if c["source_type"] in {"private", "user_upload"}]
        web_titles = [c["title"] for c in citations if c["source_type"] == "web"]
        # 证据分析先列出最多六条编号、类型和标题。
        evidence_analysis = [
            f'[{item.get("index", "?")}] {item.get("source_label", "未知来源")}：{item.get("title", "未命名来源")}'
            for item in citations[:6]
        ]
        # 用户材料单独汇总，突出它们是案件事实线索而非公共法律。
        if private_titles:
            evidence_analysis.append("用户材料：" + "、".join(private_titles[:4]))
        # 优先列公共案例；没有案例时才列少量网页资料。
        related_cases = [f"相关案例：{title}" for title in public_case_titles[:4]]
        if not related_cases:
            related_cases = [f"参考网页资料：{title}" for title in web_titles[:2]]
        # 返回结构稳定的元数据；复杂分析正文由答案本身承担。
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
        """根据问题关键词选择用户容易理解的方案标题。"""

        text = str(question or "").strip()
        # 从更具体的人身损害、婚姻、劳动、借款、合同依次判断。
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

    def stream_answer_text(
        self, question: str, evidence: list[dict], history: list[dict] | None = None,
        thinking_enabled: bool | None = None, answer_detail: str = "standard",
    ):
        """流式调用大模型；失败时自动改用非流式或本地兜底并继续逐字输出。"""

        # 按配置限制每条证据进入回答提示词的字符数。
        context = self._evidence_context(evidence, max_chars=model_setting_value(self.model, "answer_context_max_chars", 800))
        # 提示词需要知道是否包含用户材料，以便区分用户陈述和法律依据。
        has_private_material = any(row.get("source_type") in {"private", "user_upload"} for row in evidence)
        try:
            # stream_chat 产生文本增量，yield from 原样向 workflow 传递。
            yield from self._call_model(
                self.model.stream_chat,
                build_answer_messages(
                    question, context, history, has_private_material,
                    use_compact=True, answer_detail=answer_detail,
                ),
                thinking_enabled,
                answer_token_budget(model_setting_value(self.model, "answer_max_tokens", 2200), answer_detail),
                model_setting_float(self.model, "answer_llm_temperature", 0.25),
            )
        except Exception:
            # 流式接口失败后先尝试同一生成器的非流式路径。
            logger.warning("LLM 流式生成失败，准备降级到非流式或本地兜底答案", extra={"event": "llm_stream_fallback"}, exc_info=True)
            try:
                fallback = self.generate(
                    question, evidence, history,
                    thinking_enabled=False, answer_detail=answer_detail,
                ).get("answer", "")
            except Exception:
                # 非流式也失败时，根据有无证据选择本地兜底。
                citations = build_citations(evidence)
                fallback = self._fallback_answer(question, evidence, citations) if evidence else self._no_evidence_answer(question)
            # 无论来自哪条降级路线，都逐字符输出以保持前端协议一致。
            for char in str(fallback or ""):
                yield char

    @staticmethod
    def _no_evidence_answer(question: str) -> str:
        """没有任何可引用证据且模型不可用时，按常见领域返回保守本地指引。"""

        text = str(question or "").strip()
        # 人身冲突场景强调正在进行的侵害、必要限度和证据固定。
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
        # 劳动场景强调劳动关系、工资标准、欠付期间和仲裁材料。
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
        # 婚姻家庭场景强调资产清单、过错证据和取证合法性。
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
        # 其他问题使用通用民事关系、事实、证据和救济路径框架。
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
        """没有检索来源时生成领域化行动清单，同时保持 citations 为空。"""

        text = str(question or "")
        # 所有领域共享的证据缺口、空依据和空引用说明。
        common = {
            "key_issues": ["当前没有可引用来源，只能先按用户陈述作一般法律方向判断。"],
            "evidence_analysis": ["未检索到可引用的本地知识库、网页资料或用户上传材料；需要补充能对应关键事实的证据。"],
            "legal_basis": [],
            "related_cases": [],
            "citations": [],
        }
        # 劳动问题给出劳动关系、工资和仲裁材料清单。
        if any(word in text for word in ("工资", "劳动", "加班", "辞退", "离职", "工伤", "社保", "劳动合同")):
            return {
                **common,
                "action_steps": ["整理劳动关系、入离职、工资标准和欠付金额的时间线。", "保存劳动合同、工资流水、考勤、排班、聊天记录和催款记录。", "根据证据情况选择协商、投诉或劳动仲裁。"],
                "document_checklist": ["劳动合同或入职证明", "工资条、银行流水或转账记录", "考勤、排班、加班通知", "离职、辞退、催款等沟通记录"],
                "questions_to_confirm": ["是否有劳动合同或实际用工证明？", "欠付工资或补偿对应的期间和金额是多少？", "是否已经离职、被辞退或申请过仲裁？"],
                "suggestions": ["先把劳动关系和金额证据固定，再判断投诉、仲裁或诉讼路径。"],
                "risk_notice": "当前回答没有引用到检索来源，只能作为一般劳动争议信息参考；仲裁时效和金额计算需结合现行依据核验。",
            }
        # 婚姻家庭问题给出身份、财产、子女和财产转移线索清单。
        if any(word in text for word in ("离婚", "夫妻", "婚姻", "共同财产", "抚养", "彩礼")):
            return {
                **common,
                "action_steps": ["列明婚姻登记、子女、共同财产、共同债务和争议焦点。", "保存房产、车辆、存款、投资、转账、聊天记录等原始材料。", "如担心财产转移，及时固定线索并咨询是否申请调查取证或保全。"],
                "document_checklist": ["结婚证、身份证明和户籍材料", "房产、车辆、银行流水、投资或债务凭证", "子女抚养相关支出和照护证据", "对方隐匿、转移或处分财产的线索"],
                "questions_to_confirm": ["双方是否已经登记结婚、是否有子女？", "争议财产取得时间和出资来源是什么？", "是否存在隐匿、转移、变卖或挥霍共同财产的证据？"],
                "suggestions": ["婚姻财产问题先做资产清单和证据目录，不要凭猜测下结论。"],
                "risk_notice": "当前回答没有引用到检索来源，只能作为一般婚姻家庭法律信息参考；房产、子女抚养和大额财产建议结合原件咨询律师。",
            }
        # 人身冲突问题给出报警、监控、证人、伤情材料清单。
        if any(word in text for word in ("打", "动手", "反击", "正当防卫", "互殴", "争执", "伤")):
            return {
                **common,
                "action_steps": ["先固定现场证据和沟通记录。", "及时报警、就医或申请调取监控。", "后续根据处理结果考虑复议、诉讼或索赔。"],
                "document_checklist": ["报警记录或接处警回执", "监控录像或现场视频", "证人联系方式", "病历、诊断证明和费用票据", "聊天记录或其他沟通证据"],
                "questions_to_confirm": ["对方先动手的过程有没有监控或证人？", "你反击时对方侵害是否仍在持续？", "双方是否有人受伤并做过伤情鉴定？"],
                "suggestions": ["先把事实经过和证据固定下来，再判断是否构成正当防卫或防卫过当。"],
                "risk_notice": "当前回答没有引用到检索来源，只能作为一般法律信息参考；涉及处罚、伤情鉴定或刑事风险时建议尽快咨询律师。",
            }
        # 其他民事问题使用通用合同、付款、沟通和损失证据框架。
        return {
            **common,
            "action_steps": ["把争议事实按时间线整理清楚。", "保存合同、付款、沟通、通知和损失凭证等原始材料。", "根据证据情况选择协商、书面催告、调解、仲裁或起诉。"],
            "document_checklist": ["合同、协议或交易凭证", "付款、转账、收据或发票", "聊天记录、通知函或平台记录", "能证明损失或履行情况的材料"],
            "questions_to_confirm": ["双方是什么法律关系？", "争议发生的时间、金额或标的是什么？", "对方是否已经明确拒绝或提出抗辩？"],
            "suggestions": ["先补齐事实和证据，再判断具体请求和程序。"],
            "risk_notice": "当前回答没有引用到检索来源，只能作为一般法律信息参考；重大权益事项建议咨询执业律师。",
        }

    def generate(
        self, question: str, evidence: list[dict], history: list[dict] | None = None,
        thinking_enabled: bool | None = None, answer_detail: str = "standard",
    ) -> dict:
        """非流式生成完整回答字典；有无证据分别使用不同提示词和降级策略。"""

        # 先把最终证据转换成编号引用。
        citations = build_citations(evidence)
        # 有证据时构建来源上下文；没有时明确告诉模型“无有效检索证据”。
        context = self._evidence_context(evidence, max_chars=model_setting_value(self.model, "answer_context_max_chars", 800)) if evidence else "当前未提供有效检索证据。"
        # 标记是否有用户材料，防止模型把用户陈述误写成公共法律依据。
        has_private_material = any(row.get("source_type") in {"private", "user_upload"} for row in evidence)
        # ---------- 无证据生成路线 ----------
        if not evidence:
            try:
                # 使用专门的无证据提示词，要求明确不确定性和需要补充的信息。
                raw_answer = str(self._call_model(
                    self.model.chat,
                    build_no_evidence_answer_messages(
                        question, history, has_private_material, answer_detail=answer_detail,
                    ),
                    thinking_enabled,
                    answer_token_budget(model_setting_value(self.model, "no_evidence_answer_max_tokens", 1600), answer_detail),
                    model_setting_float(self.model, "no_evidence_llm_temperature", 0.2),
                ) or "").strip()
                # 清理代码围栏和 final_answer 标签。
                answer = self._clean_model_output(raw_answer, citations)
            except Exception:
                # 无证据模型也可能失败，稍后使用本地模板。
                logger.warning("无证据时 LLM 生成失败，使用本地兜底答案", extra={"event": "llm_no_evidence_fallback"}, exc_info=True)
                answer = ""
            # 模型返回空内容时按问题领域选择本地保守回答。
            if not answer:
                answer = self._no_evidence_answer(question)
            # 元数据同样明确没有 citations 和 legal_basis。
            metadata = self._no_evidence_metadata(question)
            return {
                "answer": answer,
                "case_analysis": self.build_case_analysis(question, evidence, answer, thinking_enabled=False),
                **metadata,
            }
        # ---------- 有证据生成路线 ----------
        try:
            # 构建紧凑法律回答提示词并调用普通 chat。
            raw_answer = str(self._call_model(
                self.model.chat,
                build_answer_messages(
                    question, context, history, has_private_material,
                    use_compact=True, answer_detail=answer_detail,
                ),
                thinking_enabled,
                answer_token_budget(model_setting_value(self.model, "answer_max_tokens", 2200), answer_detail),
                model_setting_float(self.model, "answer_llm_temperature", 0.25),
            ) or "").strip()
            answer = self._clean_model_output(raw_answer, citations)
        except Exception:
            # 模型服务异常时使用带证据数量和行动清单的本地兜底。
            logger.warning("LLM 非流式生成失败，使用本地兜底答案", extra={"event": "llm_sync_fallback"}, exc_info=True)
            answer = self._fallback_answer(question, evidence, citations)
        # 模型无异常却返回空内容时同样兜底。
        if not answer:
            answer = self._fallback_answer(question, evidence, citations)
        # 确保关键条号或来源术语至少出现在依据核对中。
        answer = self.ensure_source_highlights(answer, question, evidence)
        # 生成前端需要的引用、依据、类案和清单。
        metadata = self.answer_metadata(evidence)
        # 案件分析使用规则生成，不额外消耗模型调用。
        return {"answer": answer, "case_analysis": self.build_case_analysis(question, evidence, answer, thinking_enabled=False), **metadata}

    @staticmethod
    def _fallback_solution(question: str, answer: str, evidence: list[dict]) -> str:
        """第二次模型调用失败时，生成结构完整、可下载的本地 Markdown 方案。"""

        # 问题为空时使用通用标题文本。
        question_text = str(question or "本次法律咨询").strip() or "本次法律咨询"
        # answer 是第一轮正式回答，用作方案核心结论来源。
        answer_text = str(answer or "").strip()
        # 汇总最多六个不同来源标题。
        source_titles = []
        for row in evidence[:6]:
            title = str(row.get("title") or row.get("source_label") or "检索来源").strip()
            if title and title not in source_titles:
                source_titles.append(title)
        # 转成 Markdown 列表；没有来源时明确标注。
        source_text = "\n".join(f"- {title}" for title in source_titles) or "- 当前没有可稳定列出的参考来源"
        # 有证据只表示方向较明确，不表示最终法律结论确定。
        status = "🟢 方向较明确" if evidence else "🟠 需要重点补证"
        # 用第一轮回答前 360 字作为核心；没有时写事实不足说明。
        core = answer_text[:360] if answer_text else "当前信息还不足，需要先补齐事实经过和关键证据，再判断具体处理路径。"
        # 根据领域选择方案标题。
        title = AnswerGenerator.solution_title(question_text)
        # 返回固定九部分 Markdown，确保即使模型失败也能下载和打印。
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
        """进行第二次模型调用，把普通回答扩展成可打印法律处理方案。"""

        # answer 兼容结构化字典和纯字符串。
        answer_text = str(answer.get("answer", "") if isinstance(answer, dict) else answer or "").strip()
        # 方案同样使用最终证据上下文，不使用所有原始候选。
        context = self._evidence_context(evidence, max_chars=model_setting_value(self.model, "answer_context_max_chars", 800))
        # 提示词区分用户上传材料和公共依据。
        has_private_material = any(row.get("source_type") in {"private", "user_upload"} for row in evidence)
        # 记录方案生成开始时间。
        start = perf_counter()
        try:
            # 方案生成关闭思考展示，使用独立 token 和温度配置。
            raw_markdown = str(self._call_model(
                self.model.chat,
                build_solution_messages(question, answer_text, context, history, has_private_material),
                thinking_enabled=False,
                max_tokens=model_setting_value(self.model, "solution_max_tokens", 1800),
                temperature=model_setting_float(self.model, "solution_llm_temperature", 0.35),
            ) or "").strip()
            # 清理模型包装并按已有引用校验。
            markdown = self._clean_model_output(raw_markdown, build_citations(evidence))
            # 再兼容模型返回 Markdown 代码块的情况。
            if markdown.startswith("```"):
                lines = markdown.splitlines()
                if lines and lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                markdown = "\n".join(lines).strip()
            # 如果模型在一级标题前输出了说明，截掉说明，只保留正式方案。
            heading_match = re.search(r"(?m)^#\s+\S", markdown)
            if heading_match and heading_match.start() > 0:
                markdown = markdown[heading_match.start():].strip()
            # 截取标题后再统一清理一次。
            markdown = self._clean_model_output(markdown, build_citations(evidence))
        except Exception:
            # 模型失败时记录并在下方使用本地方案。
            logger.warning("LLM 法律解决方案生成失败，使用本地兜底方案", extra={"event": "llm_solution_fallback"}, exc_info=True)
            markdown = ""
        # 空内容或缺少一级标题都视为不可打印，切换固定方案。
        if not markdown:
            markdown = self._fallback_solution(question, answer_text, evidence)
        if not re.match(r"^#\s+\S", markdown):
            markdown = self._fallback_solution(question, answer_text, evidence)
        # 方案总耗时以毫秒返回。
        elapsed_ms = round((perf_counter() - start) * 1000, 2)
        # API 层会缓存并返回这份结构。
        return {
            "title": self.solution_title(question),
            "scope": "第二次模型调用生成",
            "markdown": markdown,
            "elapsed_ms": elapsed_ms,
            "citations": build_citations(evidence),
        }


# PDF 页面使用接近 A4 比例的像素画布；保存时按 150 DPI 转 PDF。
PAGE_WIDTH = 1240
PAGE_HEIGHT = 1754
# 左右和上下留白。
MARGIN_X = 105
MARGIN_Y = 100
# 正文可用宽度。
CONTENT_WIDTH = PAGE_WIDTH - (MARGIN_X * 2)
# 正文、一级标题和二级标题字号。
BODY_SIZE = 30
HEADING_SIZE = 42
SUBHEADING_SIZE = 34
# 每行文字额外间距。
LINE_GAP = 18
# 正文、弱化说明、表格边框和标题色。
TEXT_FILL = (35, 35, 35)
MUTED_FILL = (98, 100, 93)
TABLE_BORDER = (216, 224, 230)
TABLE_HEADER_BG = (217, 234, 247)
TABLE_HEADER_FILL = (41, 72, 92)
# 一级标题使用浅金背景和金色左侧强调线。
PRIORITY_BG = (255, 249, 233)
PRIORITY_ACCENT = (197, 164, 95)
# 表格单元格内边距。
TABLE_CELL_PADDING_X = 14
TABLE_CELL_PADDING_Y = 12


def _font_candidates(bold: bool = False) -> list[Path]:
    """按 Windows/Linux 常见位置列出支持中文的字体候选。"""

    # 粗体和普通字体分别使用不同文件名优先级。
    names = (
        ["msyhbd.ttc", "simhei.ttf", "NotoSansCJK-Bold.ttc", "NotoSerifCJK-Bold.ttc"]
        if bold
        else ["msyh.ttc", "simsun.ttc", "NotoSansCJK-Regular.ttc", "NotoSerifCJK-Regular.ttc"]
    )
    # Windows 字体目录和常见 Linux 字体目录。
    roots = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts/opentype/noto"),
        Path("/usr/share/fonts/truetype/noto"),
        Path("/usr/share/fonts/truetype/dejavu"),
    ]
    # 返回目录与文件名的全部组合，由加载函数按顺序检查存在性。
    return [root / name for root in roots for name in names]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """加载第一个存在的中文字体；都没有时退回 Pillow 默认字体。"""

    for path in _font_candidates(bold):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    # 默认字体可能无法完整显示中文，但保证函数仍有明确后备行为。
    return ImageFont.load_default()


def _wrap_line(text: str, font: ImageFont.ImageFont, width: int = CONTENT_WIDTH) -> list[str]:
    """按真实字体像素宽度逐字符换行，适合没有空格的中文。"""

    # 空文本保留一个空行，维护段落间距。
    if not text:
        return [""]
    # 创建 1×1 临时画布，只用来测量文字宽度。
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines: list[str] = []
    current = ""
    # 中文不能仅按空格分词，所以逐字符累积并测量。
    for char in text:
        candidate = current + char
        # 加入当前字符超出宽度时，先提交旧行。
        if current and draw.textlength(candidate, font=font) > width:
            lines.append(current)
            current = char
        else:
            current = candidate
    # 保存最后一行。
    if current:
        lines.append(current)
    return lines or [""]


def _strip_inline_markdown(text: str) -> str:
    """删除 PDF 绘制器不支持的行内 Markdown，只保留可见文字。"""

    cleaned = str(text or "")
    # 链接保留标题、删除 URL。
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cleaned)
    # 依次移除行内代码、粗体和斜体符号。
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", cleaned)
    # 合并多余空白。
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _split_table_cells(line: str) -> list[str]:
    """把 Markdown 表格行拆成单元格；非表格行返回空列表。"""

    value = str(line or "").strip()
    # 至少以 | 开头且后面还有另一个 | 才视为表格。
    if not value.startswith("|") or "|" not in value[1:]:
        return []
    # 去掉两端竖线后按中间竖线分列。
    return [cell.strip() for cell in value.strip("|").split("|")]


def _is_table_divider(line: str) -> bool:
    """识别 Markdown 表头下的 ``| --- | --- |`` 分隔行。"""

    cells = _split_table_cells(line)
    # 至少两列，且每格都是三个以上横线和可选冒号。
    return len(cells) >= 2 and all(re.match(r"^:?-{3,}:?$", cell) for cell in cells)


def _table_column_widths(column_count: int) -> list[int]:
    """按列数分配表格像素宽度，并修正整数取整误差。"""

    if column_count <= 0:
        return []
    # 三列表格使用更适合“事实/路径/重点”的比例，其他表格平均分。
    if column_count == 3:
        ratios = [0.31, 0.34, 0.35]
    else:
        ratios = [1 / column_count] * column_count
    # 比例乘正文宽度并转整数。
    widths = [int(CONTENT_WIDTH * ratio) for ratio in ratios]
    # 最后一列补上取整造成的像素差，确保总宽度恰好等于 CONTENT_WIDTH。
    widths[-1] += CONTENT_WIDTH - sum(widths)
    return widths


def _content_blocks(markdown: str) -> list[dict]:
    """把方案 Markdown 解析成标题、正文和表格绘制块。"""

    # blocks 保存按原文顺序排列的绘制指令。
    blocks: list[dict] = []
    # 预先加载正文、粗体、一级和二级标题字体，避免每行重复加载。
    body_font = _load_font(BODY_SIZE)
    body_bold_font = _load_font(BODY_SIZE, bold=True)
    heading_font = _load_font(HEADING_SIZE, bold=True)
    subheading_font = _load_font(SUBHEADING_SIZE, bold=True)
    # 统一换行并拆成原始行。
    raw_lines = str(markdown or "").replace("\r\n", "\n").split("\n")
    # 使用手动 index 是因为遇到表格时需要一次跳过多行。
    index = 0
    while index < len(raw_lines):
        raw_line = raw_lines[index]
        line = raw_line.strip()
        # 空行转换成带小间距的空正文块。
        if not line:
            blocks.append({"type": "text", "text": "", "font": body_font, "top_gap": 4, "bottom_gap": 4, "fill": TEXT_FILL})
            index += 1
            continue
        # 一级标题使用特殊金色块样式。
        if line.startswith("# "):
            blocks.append({"type": "heading", "text": _strip_inline_markdown(line[2:]), "font": heading_font, "top_gap": 26, "bottom_gap": 18})
            index += 1
            continue
        # 二、三、四级标题使用不同字号和上下间距。
        elif line.startswith("## "):
            text, font, top_gap, bottom_gap = _strip_inline_markdown(line[3:]), subheading_font, 22, 12
        elif line.startswith("### "):
            text, font, top_gap, bottom_gap = _strip_inline_markdown(line[4:]), body_bold_font, 18, 10
        elif line.startswith("#### "):
            text, font, top_gap, bottom_gap = _strip_inline_markdown(line[5:]), body_bold_font, 16, 8
        # 当前行像表格且下一行是分隔线时，解析完整 Markdown 表格。
        elif _split_table_cells(line) and index + 1 < len(raw_lines) and _is_table_divider(raw_lines[index + 1]):
            # 表头去掉行内 Markdown。
            header_cells = [_strip_inline_markdown(cell) for cell in _split_table_cells(line)]
            rows = []
            # 数据从分隔线下一行开始。
            row_index = index + 2
            while row_index < len(raw_lines):
                row_cells = _split_table_cells(raw_lines[row_index])
                # 列数不一致或不是表格行时，表格结束。
                if len(row_cells) != len(header_cells) or not row_cells:
                    break
                rows.append([_strip_inline_markdown(cell) for cell in row_cells])
                row_index += 1
            # 一个 table block 保存表头、所有行和间距。
            blocks.append({"type": "table", "header": header_cells, "rows": rows, "top_gap": 12, "bottom_gap": 18})
            # index 直接跳到表格之后，避免数据行再次当正文解析。
            index = row_index
            continue
        # 无序列表将 Markdown 符号改成可绘制圆点。
        elif line.startswith(("- ", "* ")):
            text, font, top_gap, bottom_gap = f"• {_strip_inline_markdown(line[2:])}", body_font, 4, 4
        # 有序列表保留数字开头。
        elif re.match(r"^\d+[.)]\s+", line):
            text, font, top_gap, bottom_gap = _strip_inline_markdown(line), body_font, 4, 4
        # 引用块去掉 >，并在下方使用弱化颜色。
        elif line.startswith(">"):
            text, font, top_gap, bottom_gap = _strip_inline_markdown(line.lstrip("> ")), body_font, 8, 4
        else:
            # 其余内容按普通正文处理。
            text, font, top_gap, bottom_gap = _strip_inline_markdown(line), body_font, 4, 4
        # 保存普通文字绘制块。
        blocks.append({"type": "text", "text": text, "font": font, "top_gap": top_gap, "bottom_gap": bottom_gap, "fill": MUTED_FILL if line.startswith(">") else TEXT_FILL})
        index += 1
    return blocks


def _new_pdf_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """创建一张白色 PDF 页面图像及其绘图对象。"""

    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    return page, ImageDraw.Draw(page)


def _line_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    """计算一行文字实际高度并加入行距，最低为 42 像素。"""

    bbox = draw.textbbox((0, 0), text or " ", font=font)
    return max(42, bbox[3] - bbox[1] + LINE_GAP)


def _text_block_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int = CONTENT_WIDTH) -> tuple[list[str], int]:
    """先自动换行，再返回行列表和整个文字块高度。"""

    lines = _wrap_line(text, font, width)
    return lines, sum(_line_height(draw, line, font) for line in lines)


def _draw_text_block(draw: ImageDraw.ImageDraw, y: int, block: dict) -> int:
    """从 y 坐标开始绘制普通文字块，并返回绘制结束后的 y。"""

    text = str(block.get("text") or "")
    font = block["font"]
    # 按正文宽度换行。
    lines, _height = _text_block_height(draw, text, font)
    for line in lines:
        # 所有正文从左边距开始。
        draw.text((MARGIN_X, y), line, fill=block.get("fill", TEXT_FILL), font=font)
        y += _line_height(draw, line, font)
    return y


def _draw_heading_block(draw: ImageDraw.ImageDraw, y: int, block: dict) -> int:
    """绘制带浅金背景和左侧强调线的一级标题块。"""

    text = str(block.get("text") or "")
    font = block["font"]
    # 为左侧强调线和内部留白减少 42 像素可用宽度。
    lines, text_height = _text_block_height(draw, text, font, CONTENT_WIDTH - 42)
    # 上下各留约 16 像素。
    block_height = text_height + 32
    # 绘制背景和左侧金线。
    draw.rounded_rectangle((MARGIN_X, y, PAGE_WIDTH - MARGIN_X, y + block_height), radius=4, fill=PRIORITY_BG)
    draw.rectangle((MARGIN_X, y, MARGIN_X + 8, y + block_height), fill=PRIORITY_ACCENT)
    text_y = y + 16
    # 标题文字从强调线右侧开始逐行绘制。
    for line in lines:
        draw.text((MARGIN_X + 24, text_y), line, fill=TEXT_FILL, font=font)
        text_y += _line_height(draw, line, font)
    return y + block_height


def _table_row_layout(draw: ImageDraw.ImageDraw, cells: list[str], widths: list[int], font: ImageFont.ImageFont) -> tuple[list[list[str]], int]:
    """计算表格每格换行结果，并以最高单元格决定整行高度。"""

    wrapped_cells: list[list[str]] = []
    max_height = 0
    # cells 与 widths 一一对应。
    for cell, width in zip(cells, widths):
        # 单元格文字宽度扣掉左右内边距，最少保留 24 像素。
        lines = _wrap_line(_strip_inline_markdown(cell), font, max(24, width - (TABLE_CELL_PADDING_X * 2)))
        wrapped_cells.append(lines)
        # 表格一行高度必须容纳换行最多的单元格。
        max_height = max(max_height, sum(_line_height(draw, line, font) for line in lines))
    # 再加上下内边距。
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
    """绘制一行表格的背景、边框和文字，并返回下一行 y 坐标。"""

    # 先计算每格换行和统一行高。
    wrapped_cells, row_height = _table_row_layout(draw, cells, widths, font)
    # 第一列从左边距开始。
    x = MARGIN_X
    for lines, width in zip(wrapped_cells, widths):
        # 表头可以传背景色，普通行不填充。
        if fill:
            draw.rectangle((x, y, x + width, y + row_height), fill=fill)
        # 绘制单元格边框。
        draw.rectangle((x, y, x + width, y + row_height), outline=TABLE_BORDER, width=1)
        text_y = y + TABLE_CELL_PADDING_Y
        # 在单元格内按换行结果绘制文字。
        for line in lines:
            draw.text((x + TABLE_CELL_PADDING_X, text_y), line, fill=text_fill, font=font)
            text_y += _line_height(draw, line, font)
        x += width
    # 再绘制整行上下边线，减少相邻格线不齐的视觉问题。
    draw.line((MARGIN_X, y, MARGIN_X + sum(widths), y), fill=TABLE_BORDER, width=1)
    draw.line((MARGIN_X, y + row_height, MARGIN_X + sum(widths), y + row_height), fill=TABLE_BORDER, width=1)
    return y + row_height


def render_solution_pdf(markdown: str) -> bytes:
    """把法律方案 Markdown 绘制成多页 PDF，并返回内存中的 PDF 字节。"""

    # pages 保存已经完成的页面图像。
    pages: list[Image.Image] = []
    # 创建第一页及绘图对象。
    page, draw = _new_pdf_page()
    # y 是当前纵向绘制位置，从上边距开始。
    y = MARGIN_Y

    def add_page() -> None:
        """提交当前页并创建一张新的空白页。"""

        # 需要替换外层 page/draw/y，因此声明 nonlocal。
        nonlocal page, draw, y
        pages.append(page)
        page, draw = _new_pdf_page()
        y = MARGIN_Y

    # 先把 Markdown 转为绘制块，再按顺序分页绘制。
    for block in _content_blocks(markdown):
        block_type = block.get("type")
        # 每个块可以有自己的上下间距。
        top_gap = int(block.get("top_gap", 0))
        bottom_gap = int(block.get("bottom_gap", 0))
        # 表格需要按行判断分页，并在新页重复表头。
        if block_type == "table":
            header = list(block.get("header") or [])
            rows = list(block.get("rows") or [])
            # 根据表头列数计算列宽。
            widths = _table_column_widths(len(header))
            # 无表头或无列宽的异常表格跳过。
            if not header or not widths:
                continue
            # 表头粗体，数据行普通字体。
            header_font = _load_font(BODY_SIZE, bold=True)
            body_font = _load_font(BODY_SIZE)
            # 先计算表头高度以决定是否需要换页。
            header_height = _table_row_layout(draw, header, widths, header_font)[1]
            y += top_gap
            if y + header_height > PAGE_HEIGHT - MARGIN_Y:
                add_page()
            # 绘制蓝灰色表头。
            y = _draw_table_row(draw, y, header, widths, header_font, fill=TABLE_HEADER_BG, text_fill=TABLE_HEADER_FILL)
            # 逐行绘制表格数据。
            for row in rows:
                row_height = _table_row_layout(draw, row, widths, body_font)[1]
                # 当前页放不下这一整行时，新建页面并重复表头。
                if y + row_height > PAGE_HEIGHT - MARGIN_Y:
                    add_page()
                    y = _draw_table_row(draw, y, header, widths, header_font, fill=TABLE_HEADER_BG, text_fill=TABLE_HEADER_FILL)
                y = _draw_table_row(draw, y, row, widths, body_font)
            # 表格结束后增加底部间距。
            y += bottom_gap
            continue

        # 普通/标题块取得文字和字体。
        text = str(block.get("text") or "")
        font = block["font"]
        # 一级标题还包含背景内边距，因此高度额外加 32。
        if block_type == "heading":
            lines, text_height = _text_block_height(draw, text, font, CONTENT_WIDTH - 42)
            block_height = text_height + 32
        else:
            # 普通块直接计算换行文字高度。
            lines, block_height = _text_block_height(draw, text, font)
        # 当前页放不下整个块时先换页，避免标题或短段落被切开。
        if y + top_gap + block_height > PAGE_HEIGHT - MARGIN_Y:
            add_page()
        y += top_gap
        # 标题与普通文字调用不同绘制函数。
        if block_type == "heading":
            y = _draw_heading_block(draw, y, block)
        else:
            y = _draw_text_block(draw, y, block)
        y += bottom_gap
    # 最后一页尚未提交时加入 pages。
    if page not in pages:
        pages.append(page)
    # 理论上始终至少有一页；此处提供防御性后备。
    if not pages:
        page, _draw = _new_pdf_page()
        pages.append(page)

    # 在内存中保存多页 PDF。
    output = io.BytesIO()
    # 第一页负责创建文件，append_images 附加其余页面。
    pages[0].save(output, format="PDF", resolution=150.0, save_all=True, append_images=pages[1:])
    # 返回二进制给 FastAPI Response 下载。
    return output.getvalue()
