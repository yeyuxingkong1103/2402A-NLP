# 工单15：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程
"""单题快试：几版系统提示词下，4 道图文题的表现

工单15 的助手提示词是从工单14 抄过来的，里面第 2、3 条讲的
「气流方向」「配气带孔盘」「壳体」全是**上一份专利**的零件，
对 CN100347506C 是纯噪声，先验证它是不是把模型带偏了。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from ragflow_client import RAGFlow
from run_round import build_messages

HERE = Path(__file__).parent
IDS = json.loads((HERE / ".ragflow_ids.json").read_text(encoding="utf-8"))

KB = """Here is the knowledge base:
{knowledge}
The above is the knowledge base."""

VARIANTS = {
    "V0-工单14遗留": """You are a precise assistant for Chinese industrial patent documents.
Answer strictly from the knowledge base below. If it does not contain the answer, say so.

作答规则：
1. 题目给了选项时，只回答正确选项的字母（A、B、C 或 D），不要解释。
2. 涉及附图的题目，一律按图上箭头画出的实际指向判断气流/运动方向。
3. 「气流依次经过哪些部件」：按箭头指向排出零件次序后，剔除气流只是从
   内部流过的壳体（腔体、外壳、圆锥形部分），只保留气流真正**穿过**的
   零件（如配气带孔盘这类带孔件），再作答。

""" + KB,
    "V1-只留通用规则": """You are a precise assistant for Chinese industrial patent documents.
Answer strictly from the knowledge base below. If it does not contain the answer, say so.

作答规则：
1. 题目给了选项时，只回答正确选项的字母（A、B、C 或 D），不要解释。

""" + KB,
    "V2-通用+附图优先": """You are a precise assistant for Chinese industrial patent documents.
Answer strictly from the knowledge base below. If it does not contain the answer, say so.

作答规则：
1. 题目给了选项时，只回答正确选项的字母（A、B、C 或 D），不要解释。
2. 题目提到某张附图时，以知识库中带【技术图纸】标记、且图号/页码对得上的那一段
   描述为主要依据，再结合正文里的部件说明作答。
3. 图上没画方向箭头时，按描述里的方位词（上/下/左/右、内侧/外侧、嵌套）判断位置关系。

""" + KB,
    "V3-附图优先+冲突时以正文为准": """You are a precise assistant for Chinese industrial patent documents.
Answer strictly from the knowledge base below. If it does not contain the answer, say so.

作答规则：
1. 题目给了选项时，只回答正确选项的字母（A、B、C 或 D），不要解释。
2. 题目提到某张附图时，以知识库中带【技术图纸】标记、且图号/页码对得上的那一段
   描述为主要依据。
3. 若图纸描述与正文（权利要求书、说明书）对同一编号部件的说法不完全一致，
   **以正文为准** —— 图纸描述是自动生成的，可能出现编号误读。

""" + KB,
}


def main():
    qs = json.loads((HERE / "测试问题.json").read_text(encoding="utf-8"))
    target = [q for q in qs if q["group"] != 1]          # 4 道图文题
    c = RAGFlow()
    c.load_token()

    for name, sysp in VARIANTS.items():
        c._req("PUT", f"/api/v1/chats/{IDS['chat_id']}", headers=c._h(),
               json={"dataset_ids": [IDS["ds_id"]], "prompt": {"prompt": sysp}})
        got = []
        for q in target:
            ans, _, _ = c.ask(IDS["chat_id"], build_messages(q)) if False else (None, None, None)
            break
        # ask 只返回两个值
        got = []
        for q in target:
            ans, sec = c.ask(IDS["chat_id"], build_messages(q))
            got.append(ans.strip()[:1] or "空")
        ok = sum(1 for g, q in zip(got, target) if g == q["answer"])
        marks = " ".join(f"{g}{'✓' if g == q['answer'] else '✗'}" for g, q in zip(got, target))
        print(f"  {name:<22} {marks}   4 题中 {ok} 对")


if __name__ == "__main__":
    main()
