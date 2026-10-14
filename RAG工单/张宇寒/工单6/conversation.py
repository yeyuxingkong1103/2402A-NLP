"""Small, rule-based memory for follow-up questions about two companies."""

from __future__ import annotations

import re
import time


COMPANIES = (
    "武汉兴图新科电子股份有限公司",
    "武汉力源信息技术股份有限公司",
)
ALIASES = {
    "兴图新科": COMPANIES[0],
    "力源信息": COMPANIES[1],
}
PRONOUNS = re.compile(r"这个公司|这家公司|该公司|这家企业|该企业|他们|它|(?<!其)他")


def company_in(question: str) -> str:
    for company in COMPANIES:
        if company in question:
            return company
    for alias, company in ALIASES.items():
        if alias in question:
            return company
    return ""


class ConversationManager:
    def __init__(self):
        self._sessions: dict[str, tuple[str, str, float]] = {}

    def _previous(self, session_id: str) -> tuple[str, str]:
        record = self._sessions.get(session_id)
        if not record:
            return "", ""
        company, question, updated_at = record
        if time.monotonic() - updated_at > 1800:
            self._sessions.pop(session_id, None)
            return "", ""
        return company, question

    def resolve(self, question: str, session_id: str) -> str:
        question = question.strip()
        company = company_in(question)
        old_company, old_question = self._previous(session_id)

        # “那力源信息呢？” keeps the last question's intent, changing only the company.
        if company and re.fullmatch(
            r"(?:那|那么)?(?:武汉)?(?:兴图新科(?:电子股份有限公司)?|力源信息(?:技术股份有限公司)?)(?:呢|怎么样|如何)?[？?]?",
            question,
        ):
            if not old_question:
                raise ValueError("请先提出完整问题，再追问另一家公司。")
            if old_company:
                return old_question.replace(old_company, company)
            return f"{company}：{old_question}"

        if company:
            if company not in question:
                for alias, full_name in ALIASES.items():
                    if full_name == company:
                        return question.replace(alias, company)
            return question
        if PRONOUNS.search(question):
            if not old_company:
                raise ValueError("请先说明是哪家公司，再使用‘他’或‘这个公司’追问。")
            return PRONOUNS.sub(old_company, question)
        if old_company:
            return f"{old_company}：{question}"
        return question

    def remember(self, session_id: str, question: str) -> None:
        company = company_in(question)
        if not company:
            return
        if company not in question:
            for alias, full_name in ALIASES.items():
                if full_name == company:
                    question = question.replace(alias, company)
                    break
        if len(self._sessions) >= 200 and session_id not in self._sessions:
            oldest = min(self._sessions, key=lambda key: self._sessions[key][2])
            self._sessions.pop(oldest)
        self._sessions[session_id] = (company, question, time.monotonic())
