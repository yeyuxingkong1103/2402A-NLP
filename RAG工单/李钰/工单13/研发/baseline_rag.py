# -*- coding: utf-8 -*-
"""
基线 RAG (故意不优化, 模拟真实生产瓶颈)
工单编号: 人工智能 NLP-RAG 项目-RAG 性能瓶颈识别与优化

用于: 跑出 "优化前" 数据, 识别瓶颈
"""
import os, sys, time, logging, hashlib, re
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v13 as config
from performance_monitor import TraceContext, record_metric

logger = logging.getLogger(__name__)


class BaselineRAG:
    """基线 RAG - 故意慢, 模拟典型瓶颈"""

    def __init__(self, trace: TraceContext = None):
        self.trace = trace

    # ============ 5 个阶段 (串行, 不缓存, 大 Top-K) ============

    def query_process(self, query: str) -> List[str]:
        """
        瓶颈 A: Query 处理慢
        - 调用大模型做 rewrite (模拟 sleep)
        - 无缓存
        """
        if self.trace:
            self.trace.enter_stage("query_process")
        start = time.time()

        # 故意慢: 模拟 LLM 做 query rewrite
        time.sleep(config.BASELINE_SLEEP_QUERY)

        result = [query, f"rewrite:{query}"]  # 伪 rewrite

        elapsed = (time.time() - start) * 1000
        if self.trace:
            self.trace.exit_stage(f"rewrite 完成, {len(result)} 个子 query")
        record_metric("stage.query_process", elapsed)
        return result

    def retrieve(self, queries: List[str]) -> List[Dict]:
        """
        瓶颈 B: 检索慢
        - 全量扫描 (无索引)
        - Top-K=10 (过大)
        - 串行处理每个 sub-query
        """
        if self.trace:
            self.trace.enter_stage("retrieve")
        start = time.time()

        # 故意慢: 模拟全量向量库扫描
        time.sleep(config.BASELINE_SLEEP_RETRIEVE)

        # 生成伪检索结果 (Top-10)
        chunks = []
        for i in range(config.BASELINE_TOP_K):
            chunks.append({
                "id": f"chunk_{i}",
                "text": f"基线检索结果 #{i}: 关于'{queries[0][:20]}'的相关段落, 包含关键词...",
                "score": 10.0 - i * 0.8,
                "source": f"doc_{i % 3}.pdf",
            })

        elapsed = (time.time() - start) * 1000
        if self.trace:
            self.trace.exit_stage(f"检索到 {len(chunks)} 个 chunk")
        record_metric("stage.retrieve", elapsed)
        return chunks

    def context_assemble(self, chunks: List[Dict], query: str) -> str:
        """
        瓶颈 C: 上下文组装慢
        - 无去重
        - 无长度限制
        - 串行拼接
        """
        if self.trace:
            self.trace.enter_stage("context_assemble")
        start = time.time()

        # 故意慢: 复杂格式 + 多次遍历
        time.sleep(config.BASELINE_SLEEP_CONTEXT)

        # 模拟复杂拼接 (去重 + 排序 + 格式化)
        seen_ids = set()
        sorted_chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)
        parts = []
        for c in sorted_chunks:
            if c["id"] in seen_ids:
                continue
            seen_ids.add(c["id"])
            parts.append(f"[{c['source']}] {c['text']}")

        context = "\n\n".join(parts)
        prompt = (
            f"基于以下资料回答问题, 用专业术语, 引用来源:\n\n"
            f"【资料】\n{context}\n\n【问题】{query}\n\n【回答】"
        )

        elapsed = (time.time() - start) * 1000
        if self.trace:
            self.trace.exit_stage(f"prompt 长度={len(prompt)} chars")
        record_metric("stage.context_assemble", elapsed)
        return prompt

    def llm_generate(self, prompt: str) -> str:
        """
        瓶颈 D: LLM 生成慢 (通常最重)
        - 大模型 (GPT-4 级别)
        - 长上下文
        - 无流式
        """
        if self.trace:
            self.trace.enter_stage("llm_generate")
        start = time.time()

        # 故意慢: 模拟大 LLM API 延迟
        time.sleep(config.BASELINE_SLEEP_LLM)

        # 基于预设答案的伪生成
        answer = self._mock_generate(prompt)

        elapsed = (time.time() - start) * 1000
        if self.trace:
            self.trace.exit_stage(f"生成 {len(answer)} chars")
        record_metric("stage.llm_generate", elapsed)
        return answer

    def post_process(self, answer: str, chunks: List[Dict]) -> Dict:
        """
        瓶颈 E: 后处理慢
        - 复杂引用校验
        - 格式化
        - 同步阻塞
        """
        if self.trace:
            self.trace.enter_stage("post_process")
        start = time.time()

        # 故意慢: 模拟复杂校验
        time.sleep(config.BASELINE_SLEEP_POST)

        sources = list({c["source"] for c in chunks})
        result = {
            "answer": answer,
            "sources": sources,
            "num_chunks": len(chunks),
            "format_version": "1.0",
        }

        elapsed = (time.time() - start) * 1000
        if self.trace:
            self.trace.exit_stage(f"引用 {len(sources)} 个来源")
        record_metric("stage.post_process", elapsed)
        return result

    # ============ 端到端 ============

    def query(self, query: str) -> Dict:
        """端到端基线 RAG"""
        t0 = time.time()

        sub_queries = self.query_process(query)
        chunks = self.retrieve(sub_queries)
        prompt = self.context_assemble(chunks, query)
        answer = self.llm_generate(prompt)
        result = self.post_process(answer, chunks)

        total_ms = (time.time() - t0) * 1000
        record_metric("end_to_end.baseline", total_ms)
        result["total_ms"] = round(total_ms, 2)
        result["mode"] = "Baseline RAG (未优化)"
        return result

    # ============ 内部 ============

    def _mock_generate(self, prompt: str) -> str:
        """从 prompt 生成伪回答"""
        # 提取问题
        q_match = re.search(r"【问题】(.+)", prompt)
        question = q_match.group(1).strip() if q_match else ""

        # 预设答案库
        answers = {
            "控股股东": "武汉力源科技有限公司是武汉力源信息技术股份有限公司的控股股东, 持股比例 35%。",
            "注册资本": "武汉兴图新科电子股份有限公司的注册资本为 7360 万元。",
            "募集资金": "募集资金拟投资 ADC/DAC 芯片、高速接口芯片、射频前端芯片及补充流动资金。",
            "发行股数": "本次发行股数为 2000 万股, 占发行后总股本的比例为 25.00%。",
            "销售部": "销售部有 4 个部门: 大客户销售部、渠道销售部、产品销售部、电商销售部; 其中大客户销售部有 4 个销售处: 华东、华南、华北、西南销售处。",
            "军用领域": "军用领域收入分别为 6464.51 万元、14414.16 万元、18780.67 万元、4627.14 万元; 占主营业务收入比重分别为 82.10%、97.31%、94.84%、94.34%。",
            "IC 市场": "增长率最快的是汽车电子和嵌入式系统行业; 负增长的是消费电子和通讯设备行业。",
            "技术标准": "武汉兴图新科电子股份有限公司参与制定了 AVS 编解码技术标准。",
        }

        for kw, ans in answers.items():
            if kw in question:
                return ans

        # 从检索 chunk 里拼一段
        chunks = re.findall(r"基线检索结果 #(\d+): (.+?)(?=\n|$)", prompt)
        if chunks:
            return "根据检索结果, " + chunks[0][1][:100]
        return "未能生成有效回答"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trace = TraceContext()
    rag = BaselineRAG(trace)

    q = "武汉力源的控股股东是谁?"
    result = rag.query(q)
    trace.save_trace()

    print(f"\n=== Baseline RAG ===")
    print(f"问题: {q}")
    print(f"回答: {result['answer']}")
    print(f"总耗时: {result['total_ms']:.0f} ms  ({result['total_ms']/1000:.2f}s)")
    print(f"\n阶段明细:")
    for s in trace.stages:
        print(f"  {s['name']:<20} {s['elapsed_ms']:>8.1f} ms")

    bottleneck = trace._find_bottleneck()
    print(f"\n⚠️  最大瓶颈: {bottleneck['name']} ({bottleneck['elapsed_ms']:.1f}ms)")
