"""工单编号：人工智能NLP-RAG-Query理解优化任务。"""

import re


def company_name(text):
    text = re.sub(r"^(那|那么)", "", text.strip())
    matches = re.findall(r"(?:^|[，,。；;：:\s])([\u4e00-\u9fffA-Za-z0-9（）()·]{2,40}?(?:股份有限公司|有限公司))", text)
    return matches[-1] if matches else ""


def resolve_query(question, history):
    """把代词和“那……呢”补成可独立检索的问题。"""
    question = question.strip()
    explicit_company = company_name(question)
    previous = history[-1].get("resolved_question", "") if history else ""
    previous_company = company_name(previous)
    company = explicit_company or next(
        (company_name(item.get("resolved_question", "")) for item in reversed(history)
         if company_name(item.get("resolved_question", ""))), ""
    )

    if explicit_company and re.match(r"^(那|那么)", question) and re.search(r"呢[？?]?$", question) and previous:
        if previous_company:
            return previous.replace(previous_company, explicit_company)
        return explicit_company + question[question.find("呢") + 1:]

    for pronoun in ("这个公司", "这家公司", "该公司", "他", "她", "它"):
        if pronoun in question and company:
            question = question.replace(pronoun, company)

    if not explicit_company and company and re.search(r"呢[？?]?$", question) and previous:
        topic = re.sub(r"[？?。]$", "", previous)
        return topic + " " + question[:-1]
    return question
