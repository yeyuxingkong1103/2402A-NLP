# -*- coding: utf-8 -*-
"""评测管线：对每题跑真实 RAG 流程，产出 RAGAS 所需的样本字段。"""
from app.core.prompts import build_messages
# 解析：提示词拼装（评测与生产同一条链路）


# 评测管线：检索 → 带知识生成 → 组装 RAGAS 样本字段
class EvalPipeline:
    # 解析：评测管线
    def __init__(self, knowledge_service, llm, role, top_k: int = 4, recall_k: int = 30):
        # 解析：构造——注入真实组件（评测走真检索真生成）
        self.knowledge_service = knowledge_service
        # 解析：知识库服务
        self.llm = llm
        # 解析：大模型
        self.role = role
        # 解析：评测角色
        self.top_k = top_k
        # 解析：注入条数
        self.recall_k = recall_k
        # 解析：召回条数

    async def run_sample(self, sample: dict) -> dict:
        """单题：混合检索+重排 → 带知识生成 → 组装 ragas 样本。"""
        question = sample["question"]
        # 解析：取题目
        contexts = await self.knowledge_service.retrieve(
            # 解析：真实检索
            self.role.id, question, top_k=self.top_k, recall_k=self.recall_k
            # 解析：角色、问题、Top、召回
        )
        knowledge = "\n\n".join(contexts)
        # 解析：知识块拼成文本
        messages = build_messages(
            # 解析：拼提示词（与生产完全一致）
            role_name=self.role.name,
            # 解析：角色名
            persona=self.role.persona,
            # 解析：人设
            prompt_template=self.role.prompt_template,
            # 解析：模板
            history=[],
            # 解析：评测无多轮历史
            user_input=question,
            # 解析：题目作为输入
            knowledge=knowledge,
            # 解析：检索知识
        )
        answer = await self.llm.chat(messages)
        # 解析：真实生成
        return {
            # 解析：组装 RAGAS 五字段样本
            "user_input": question,
            # 解析：问题
            "response": answer,
            # 解析：生成回答
            "retrieved_contexts": contexts,
            # 解析：检索上下文
            "reference": sample["reference_answer"],
            # 解析：参考答案
            "reference_contexts": sample["reference_contexts"],
            # 解析：标准上下文（算 recall 用）
        }

    async def run_all(self, samples: list[dict]) -> list[dict]:
        # 解析：逐题跑全部
        return [await self.run_sample(sample) for sample in samples]
        # 解析：顺序执行返回全部样本结果
