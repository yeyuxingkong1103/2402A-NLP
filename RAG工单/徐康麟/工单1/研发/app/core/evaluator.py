"""评估：RAG vs 纯 LLM 对比、RAGAS 指标、确定性质量指标。

工单要求（第 8 节）：
1. 对 10 个固定问题分别运行 RAG 回答与纯 LLM 回答（不读 PDF）；
2. 使用 RAGAS 评估 faithfulness / answer_relevancy / context_precision / context_recall；
3. 生成对比报告 ``优化/评估结果/eval_results/rag_vs_llm.csv`` 与 ``ragas_report.md``；
4. 评估指标包括：准确率、首字响应时间、引用正确率、“不清楚”回复正确率；
5. 评估脚本 ``scripts/evaluate.py``。

设计原则：**指标分两层**
- 第一层「确定性指标」：不需 LLM，任何环境都能跑，且结果可复现——
  答案准确率（数字+关键实体匹配）、引用正确率、首字延迟、“不清楚”正确率；
- 第二层「RAGAS 指标」：需要 LLM 作为裁判（可用 vLLM 提供的 OpenAI 兼容接口），
  不可用时明确标注为“未运行”，绝不伪造数值。
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from app.core.config import get_settings
from app.core.logging_conf import logger, trace
from app.core.number_utils import normalize_number_strings
from app.models.schemas import Answer, EvalRecord, GoldenQA
from app.storage.sqlite_manager import SQLiteManager, get_sqlite_manager

try:  # pragma: no cover
    from ragas import EvaluationDataset, SingleTurnSample, evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    HAS_RAGAS = True
except Exception:  # pragma: no cover
    HAS_RAGAS = False
    EvaluationDataset = None  # type: ignore
    SingleTurnSample = None  # type: ignore
    ragas_evaluate = None  # type: ignore

NUMBER_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
PERCENT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%")
# 答案中的“关键实体”：公司/工程/标准名称等
ENTITY_PATTERN = re.compile(r"[“\"]([^”\"]{4,60})[”\"]")
# 归一化时丢弃的标点（中英文标点均含）
PUNCT_TO_STRIP = "，,。.、；;：:！!？?（）()【】[]《》〈〉\"'“”‘’ \t\n\r\u3000"
# 字符二元组相似度阈值：低于该值判定为“答非所问”
FUZZY_THRESHOLD = 0.62


def _bigrams(text: str) -> set[str]:
    """字符二元组集合（中文无需分词即可度量字面重叠）。"""
    cleaned = "".join(char for char in text if char not in PUNCT_TO_STRIP)
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


# 任意数值（含百分比、整数、小数），用于“是否漏答了某个数字”的判定
ALL_NUMBER_PATTERN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
# 百分比（值 + 百分号），用于比例类答案的完整度检查
PERCENT_ONLY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _bigram_number(raw: str) -> str:
    """把数字字面量规范化，便于集合比较（去掉千分位与多余小数零）。"""
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return raw
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass
class MetricSummary:
    """一组评估结果的汇总指标。"""

    mode: str
    count: int = 0
    correct: int = 0
    unknown_count: int = 0
    unknown_expected: int = 0
    unknown_correct: int = 0
    citation_total: int = 0
    citation_valid: int = 0
    first_token_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return round(self.correct / self.count, 4) if self.count else 0.0

    @property
    def unknown_accuracy(self) -> float:
        """“不清楚”回复正确率：该拒答时拒答、不该拒答时没拒答。"""
        return round(self.unknown_correct / self.count, 4) if self.count else 0.0

    @property
    def citation_accuracy(self) -> float:
        return round(self.citation_valid / self.citation_total, 4) if self.citation_total else 0.0

    @property
    def first_token_avg_ms(self) -> float:
        return round(statistics.mean(self.first_token_ms), 2) if self.first_token_ms else 0.0

    @property
    def first_token_max_ms(self) -> float:
        return round(max(self.first_token_ms), 2) if self.first_token_ms else 0.0

    @property
    def first_token_p95_ms(self) -> float:
        if not self.first_token_ms:
            return 0.0
        ordered = sorted(self.first_token_ms)
        index = min(int(len(ordered) * 0.95), len(ordered) - 1)
        return round(ordered[index], 2)

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "count": self.count,
            "accuracy": self.accuracy,
            "unknown_accuracy": self.unknown_accuracy,
            "citation_accuracy": self.citation_accuracy,
            "citation_total": self.citation_total,
            "citation_valid": self.citation_valid,
            "first_token_avg_ms": self.first_token_avg_ms,
            "first_token_max_ms": self.first_token_max_ms,
            "first_token_p95_ms": self.first_token_p95_ms,
            "total_avg_ms": round(statistics.mean(self.total_ms), 2) if self.total_ms else 0.0,
        }


class Evaluator:
    """评估器。"""

    def __init__(self, store: SQLiteManager | None = None) -> None:
        self.settings = get_settings()
        self.store = store or get_sqlite_manager()
        self.results_dir = self.settings.paths.eval_results
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 确定性判分
    # ==================================================================
    @staticmethod
    def _normalize(text: str) -> str:
        """归一化：去空白、统一逗号与括号，便于稳健比对。"""
        if not text:
            return ""
        text = text.replace("，", ",").replace("（", "(").replace("）", ")")
        text = text.replace("％", "%").replace(" ", "").replace("\u3000", "")
        return text.strip()

    def check_answer(self, answer: str, golden: str) -> tuple[bool, str]:
        """判断回答是否正确（确定性规则，不依赖 LLM）。

        判定顺序（满足任一即算对）：
        1. 去掉标点与空白后，参考答案是答案的子串（或反之）；
        2. 参考答案中的**所有数字**都出现在答案里；
        3. 双方关键实体（引号内名称）有交集；
        4. 字符二元组 Jaccard 相似度 ≥ ``FUZZY_THRESHOLD``
           （容忍“军队视频指挥”与“国防军队视频指挥”这类同义改写）。

        说明：第 4 条是为了不把**语义正确但措辞略有差异**的回答误判为错，
        阈值取得较保守，且必须在数字/实体都未命中时才生效。

        Returns:
            ``(是否正确, 判分说明)``
        """
        if not answer or not golden:
            return False, "答案或参考答案为空"
        norm_answer, norm_golden = self._normalize(answer), self._normalize(golden)

        # 1. 去标点的包含关系
        plain_answer = "".join(char for char in norm_answer if char not in PUNCT_TO_STRIP)
        plain_golden = "".join(char for char in norm_golden if char not in PUNCT_TO_STRIP)
        if plain_golden and plain_golden in plain_answer:
            return True, "参考答案为答案子串（忽略标点）"
        if plain_answer and plain_answer in plain_golden:
            return True, "答案为参考答案子串（忽略标点）"

        # 2. 金额全部命中（比较折算成元后的规范值，容忍 5,520 与 5,520.00 万元
        #    这类等价写法，也容忍 1.5 亿元 与 15,000 万元 这类跨单位写法）
        golden_amounts = normalize_number_strings(norm_golden)
        answer_amounts = normalize_number_strings(norm_answer)
        if golden_amounts and golden_amounts.issubset(answer_amounts):
            return True, f"金额全部命中({len(golden_amounts)}个)"

        # 2.1 跨单位写法可能拆出多个候选值（如 1.5 亿元 -> 1.5e8 与裸数字），
        #     只要参考答案的每个金额都能在答案里找到**相等的**金额，即算命中。
        if golden_amounts and all(
            any(abs(float(got) - float(want)) <= max(1.0, abs(float(want))) * 1e-6 for got in answer_amounts)
            for want in golden_amounts
        ):
            return True, f"金额等价命中({len(golden_amounts)}个)"

        # 3. 关键实体命中
        golden_entities = set(ENTITY_PATTERN.findall(golden))
        answer_entities = set(ENTITY_PATTERN.findall(answer))
        if golden_entities and golden_entities & answer_entities:
            return True, "关键实体命中"

        # 4. 比例单独兜底
        golden_percents = {pct.replace(" ", "") for pct in PERCENT_PATTERN.findall(norm_golden)}
        answer_percents = {pct.replace(" ", "") for pct in PERCENT_PATTERN.findall(norm_answer)}
        if golden_percents and golden_percents.issubset(answer_percents):
            return True, "比例全部命中"

        # 5. 模糊相似度：**必须在不缺任何数字的前提下**才启用。
        #
        #    这里必须看**全部数字**（含百分比），不能只看金额：
        #    参考答案是 4 个比重，回答只给出 2 个时文字高度重合（相似度 0.68），
        #    若只比对金额就会漏放行，把“漏答一半”判成正确，准确率虚高。
        missing_amounts = golden_amounts - answer_amounts
        golden_all = {_bigram_number(match) for match in ALL_NUMBER_PATTERN.findall(norm_golden)}
        answer_all = {_bigram_number(match) for match in ALL_NUMBER_PATTERN.findall(norm_answer)}
        missing_all = golden_all - answer_all
        if not missing_amounts and not missing_all:
            similarity = _jaccard(_bigrams(norm_answer), _bigrams(norm_golden))
            if similarity >= FUZZY_THRESHOLD:
                return True, f"字符二元组相似度 {similarity:.2f} ≥ {FUZZY_THRESHOLD}"

        missing = sorted(missing_all) or sorted(missing_amounts)
        return False, f"未命中：缺少数值 {missing[:6]}" if missing else "未命中：文本与数值均不匹配"

    @trace
    def evaluate_answer(self, golden: GoldenQA, answer: Answer, mode: str) -> EvalRecord:
        """对单条回答打分并生成 ``EvalRecord``。"""
        is_correct, note = self.check_answer(answer.answer, golden.answer)
        if golden.should_be_unknown:
            is_correct = answer.is_unknown
            note = "该问题应回复不清楚" if answer.is_unknown else "该问题应回复不清楚，但模型给出了答案"

        citation_pages = [citation.page for citation in answer.citations]
        # 逐条判合法：必须带真实来源片段（chunk_id），页码缺失或臆造的引用不算数。
        # 注意不能只用“整题的布尔值”，否则 5 条引用里 1 条有效会被算成整题有效。
        valid_count = sum(1 for citation in answer.citations if citation.chunk_id and citation.page > 0)
        citation_valid = bool(citation_pages) and valid_count == len(citation_pages)

        record = EvalRecord(
            question_id=golden.id,
            question=golden.question,
            mode=mode,  # type: ignore[arg-type]
            answer=answer.answer,
            golden=golden.answer,
            is_correct=is_correct,
            is_unknown=answer.is_unknown,
            should_be_unknown=golden.should_be_unknown,
            citation_pages=citation_pages,
            citation_valid=citation_valid,
            citation_valid_count=valid_count,
            first_token_ms=answer.first_token_ms,
            total_ms=answer.total_ms,
        )
        logger.info(
            "app.core.evaluator",
            "单题评估完成",
            question_id=golden.id,
            mode=mode,
            is_correct=is_correct,
            note=note,
            citation_total=len(citation_pages),
            citation_valid=valid_count,
        )
        return record

    # ==================================================================
    # 汇总
    # ==================================================================
    def summarize(self, records: Iterable[EvalRecord]) -> dict[str, MetricSummary]:
        """按模式汇总指标。"""
        summaries: dict[str, MetricSummary] = {}
        for record in records:
            summary = summaries.setdefault(record.mode, MetricSummary(mode=record.mode))
            summary.count += 1
            summary.correct += int(record.is_correct)
            summary.unknown_count += int(record.is_unknown)
            if record.should_be_unknown:
                summary.unknown_expected += 1
            # “不清楚”回复正确率：该拒答则拒答，不该拒答则必须给出答案
            if record.should_be_unknown == record.is_unknown:
                summary.unknown_correct += 1
            summary.citation_total += len(record.citation_pages)
            summary.citation_valid += record.citation_valid_count
            summary.first_token_ms.append(record.first_token_ms)
            summary.total_ms.append(record.total_ms)
        return summaries

    # ==================================================================
    # 报告输出
    # ==================================================================
    @trace
    def write_comparison_csv(self, records: list[EvalRecord], filename: str = "rag_vs_llm.csv") -> Path:
        """输出 RAG vs 纯 LLM 的逐题对比 CSV。"""
        target = self.results_dir / filename
        by_question: dict[int, dict[str, EvalRecord]] = {}
        for record in records:
            by_question.setdefault(record.question_id, {})[record.mode] = record

        with open(target, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "question_id",
                    "question",
                    "golden",
                    "rag_answer",
                    "rag_correct",
                    "rag_citation_pages",
                    "rag_first_token_ms",
                    "rag_total_ms",
                    "llm_answer",
                    "llm_correct",
                    "answer_diff",
                ]
            )
            for question_id in sorted(by_question):
                modes = by_question[question_id]
                rag: EvalRecord | None = modes.get("rag") or modes.get("extractive")
                llm: EvalRecord | None = modes.get("llm")
                writer.writerow(
                    [
                        question_id,
                        (rag or llm).question if (rag or llm) else "",
                        (rag or llm).golden if (rag or llm) else "",
                        rag.answer if rag else "",
                        int(rag.is_correct) if rag else "",
                        json.dumps(rag.citation_pages, ensure_ascii=False) if rag else "",
                        rag.first_token_ms if rag else "",
                        rag.total_ms if rag else "",
                        llm.answer if llm else "",
                        int(llm.is_correct) if llm else "",
                        ("一致" if rag and llm and rag.answer.strip() == llm.answer.strip() else "不一致")
                        if rag and llm
                        else "",
                    ]
                )
        logger.info("app.core.evaluator", "对比 CSV 已生成", path=str(target), rows=len(by_question))
        return target

    @trace
    def write_report(
        self,
        records: list[EvalRecord],
        ragas_result: dict[str, object] | None = None,
        filename: str = "ragas_report.md",
        extra: dict[str, object] | None = None,
    ) -> Path:
        """输出 Markdown 评估报告。"""
        target = self.results_dir / filename
        summaries = self.summarize(records)
        rag = summaries.get("rag") or summaries.get("extractive")
        llm = summaries.get("llm")

        lines: list[str] = []
        lines.append("# RAG vs 纯 LLM 评估报告\n")
        lines.append("> 本报告由 `scripts/evaluate.py` 自动生成。")
        lines.append("> 所有数值均来自本次实际运行；未运行的指标会明确标注为「未运行」，不做估算。\n")

        if extra:
            lines.append("## 0. 运行环境与被测对象\n")
            lines.append("| 项目 | 值 |")
            lines.append("| --- | --- |")
            for key, value in extra.items():
                lines.append(f"| {key} | {value} |")
            lines.append("")

        lines.append("## 1. 确定性指标（不依赖 LLM，可复现）\n")
        lines.append("| 指标 | RAG 系统 | 纯 LLM（无资料） |")
        lines.append("| --- | --- | --- |")
        rows = [
            ("题目数", "count"),
            ("答案准确率", "accuracy"),
            ("“不清楚”回复正确率", "unknown_accuracy"),
            ("引用正确率", "citation_accuracy"),
            ("首字响应时间 平均(ms)", "first_token_avg_ms"),
            ("首字响应时间 P95(ms)", "first_token_p95_ms"),
            ("首字响应时间 最大(ms)", "first_token_max_ms"),
            ("端到端 平均(ms)", "total_avg_ms"),
        ]
        rag_dict = rag.as_dict() if rag else {}
        llm_dict = llm.as_dict() if llm else {}
        for label, key in rows:
            left = rag_dict.get(key, "—")
            right = llm_dict.get(key, "—")
            lines.append(f"| {label} | {left} | {right} |")
        lines.append("")

        if rag:
            lines.append(f"- 首字响应时间预算：{self.settings.app.first_token_budget_seconds} 秒；")
            verdict = "满足" if rag.first_token_max_ms < self.settings.app.first_token_budget_seconds * 1000 else "不满足"
            lines.append(f"  本次实测最大首字延迟 {rag.first_token_max_ms} ms，**{verdict}**该预算。")
            lines.append("")

        lines.append("## 2. RAGAS 指标\n")
        if ragas_result:
            lines.append("| 指标 | 数值 |")
            lines.append("| --- | --- |")
            for key, value in ragas_result.items():
                lines.append(f"| {key} | {value} |")
        else:
            lines.append("**未运行。** RAGAS 的四项指标（faithfulness / answer_relevancy /")
            lines.append("context_precision / context_recall）需要 LLM 作为裁判。请在算力云上：\n")
            lines.append("```bash")
            lines.append("# 1) 先起 vLLM 服务（见 scripts/run_vllm.sh）")
            lines.append("# 2) 再带 --ragas 运行评估")
            lines.append("python scripts/evaluate.py --ragas")
            lines.append("```")
        lines.append("")

        lines.append("## 3. 逐题结果\n")
        lines.append("| 题号 | 问题 | 参考答案 | RAG 回答 | 正确 | 引用页码 | 首字(ms) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for record in sorted(records, key=lambda item: (item.question_id, item.mode)):
            if record.mode not in {"rag", "extractive"}:
                continue
            question = record.question.replace("|", "\\|")[:40]
            golden = record.golden.replace("|", "\\|")[:30]
            answer = record.answer.replace("|", "\\|").replace("\n", " ")[:60]
            mark = "✅" if record.is_correct else "❌"
            pages = ",".join(str(page) for page in record.citation_pages) or "—"
            lines.append(
                f"| {record.question_id} | {question} | {golden} | {answer} | {mark} | {pages} | {record.first_token_ms} |"
            )
        lines.append("")

        if llm:
            lines.append("## 4. 纯 LLM 逐题结果（不提供任何资料）\n")
            lines.append("| 题号 | 回答 | 正确 |")
            lines.append("| --- | --- | --- |")
            for record in sorted(records, key=lambda item: item.question_id):
                if record.mode != "llm":
                    continue
                answer = record.answer.replace("|", "\\|").replace("\n", " ")[:80]
                mark = "✅" if record.is_correct else "❌"
                lines.append(f"| {record.question_id} | {answer} | {mark} |")
            lines.append("")

        lines.append("## 5. 结论\n")
        if rag and llm:
            lines.append(
                f"- RAG 准确率 {rag.accuracy:.1%}，纯 LLM 准确率 {llm.accuracy:.1%}，"
                f"差距 {rag.accuracy - llm.accuracy:+.1%}。"
            )
            lines.append(
                "- 该对比说明：招股书中的具体数字与专有名称无法由模型参数记忆获得，"
                "必须依赖文档检索。"
            )
        elif rag:
            lines.append(f"- RAG 准确率 {rag.accuracy:.1%}（未运行纯 LLM 对照组）。")
        lines.append("- 未运行的指标已在文中明确标注，不代表通过。")

        target.write_text("\n".join(lines), encoding="utf-8")
        logger.info("app.core.evaluator", "评估报告已生成", path=str(target))
        return target

    @trace
    def write_json(self, records: list[EvalRecord], filename: str = "eval_records.json") -> Path:
        """输出机器可读的完整评估记录。"""
        target = self.results_dir / filename
        payload = {
            "summaries": {mode: summary.as_dict() for mode, summary in self.summarize(records).items()},
            "records": [record.model_dump(mode="json") for record in records],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("app.core.evaluator", "评估明细已生成", path=str(target), records=len(records))
        return target

    # ==================================================================
    # RAGAS
    # ==================================================================
    def ragas_available(self) -> bool:
        return HAS_RAGAS

    @trace
    def run_ragas(
        self,
        samples: list[dict[str, object]],
        llm_base_url: str | None = None,
        model: str | None = None,
    ) -> dict[str, object] | None:
        """运行 RAGAS 四项指标。

        Args:
            samples: 每项含 ``user_input / response / retrieved_contexts / reference``。
            llm_base_url: 裁判 LLM 的 OpenAI 兼容地址（默认取配置）。
            model: 裁判模型名。

        Returns:
            指标字典；RAGAS 不可用或执行失败时返回 ``None``（并记录原因）。
        """
        if not HAS_RAGAS:
            logger.warning("app.core.evaluator", "未安装 ragas，跳过 RAGAS 评估")
            return None
        if not samples:
            logger.warning("app.core.evaluator", "RAGAS 样本为空，跳过评估")
            return None

        base_url = llm_base_url or self.settings.llm.base_url
        judge_model = model or self.settings.llm.model
        try:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_community.embeddings import HuggingFaceEmbeddings

            judge = LangchainLLMWrapper(
                ChatOpenAI(
                    base_url=base_url,
                    api_key=self.settings.llm.api_key,
                    model=judge_model,
                    temperature=0,
                )
            )
            embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name=self.settings.embedding.model_name)
            )
            dataset = EvaluationDataset(
                samples=[
                    SingleTurnSample(
                        user_input=str(sample["user_input"]),
                        response=str(sample["response"]),
                        retrieved_contexts=[str(context) for context in sample.get("retrieved_contexts", [])],
                        reference=str(sample.get("reference", "")),
                    )
                    for sample in samples
                ]
            )
            result = ragas_evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=judge,
                embeddings=embeddings,
                raise_exceptions=False,
            )
            scores = {
                key: round(float(value), 4)
                for key, value in dict(result).items()
                if isinstance(value, (int, float)) and key != "num_samples"
            }
            logger.info("app.core.evaluator", "RAGAS 评估完成", model=judge_model, scores=scores)
            return scores
        except Exception as exc:
            logger.exception(
                "app.core.evaluator", "RAGAS 评估失败（通常因为裁判 LLM 不可用）", error=f"{type(exc).__name__}: {exc}"
            )
            return None


def get_evaluator(store: SQLiteManager | None = None) -> Evaluator:
    """工厂函数。"""
    return Evaluator(store=store)
