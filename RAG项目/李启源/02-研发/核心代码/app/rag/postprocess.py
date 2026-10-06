"""Validate, sanitize, and normalize model responses."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PostProcessResult:
    """Result of the response post-processing pipeline."""

    text: str
    replacements_made: int
    sensitive_words_found: int
    format_issues_fixed: int
    is_valid: bool
    warnings: list[str]


class SensitiveWordFilter:
    """Mask configured sensitive terms without changing surrounding text."""

    def __init__(self, sensitive_words: list[str] | None = None) -> None:
        words = sensitive_words or ["身份证号", "手机号码", "银行卡号", "密码", "骗子", "诈骗"]
        self.patterns = [re.compile(re.escape(word), re.IGNORECASE) for word in words]

    def filter(self, text: str, mask_char: str = "*") -> tuple[str, int]:
        count = 0
        for pattern in self.patterns:
            matches = pattern.findall(text)
            count += len(matches)
            text = pattern.sub(lambda match: mask_char * len(match.group(0)), text)
        if count:
            logger.warning("Filtered %s sensitive words from output", count)
        return text, count


class RegexReplacer:
    """Apply the small normalization set used by generated answers."""

    def __init__(self, replacement_rules: list[tuple[str, str]] | None = None) -> None:
        self.rules = replacement_rules or [
            (r"\s+", " "),
            (r"([。！？,，])\1+", r"\1"),
            (r"您您", "您"),
            (r'"', '"'),
            (r"'", "'"),
            (r"(?m)^\s+", ""),
            (r"(?m)\s+$", ""),
        ]

    def apply(self, text: str) -> tuple[str, int]:
        count = 0
        for pattern, replacement in self.rules:
            text, changed = re.subn(pattern, replacement, text)
            count += changed
        return text, count


class FormatValidator:
    """Validate and repair length, ending punctuation, and repetition."""

    _ENDING_PUNCT = '。！？.!?…~～)）】」』"\''
    _EMOJI = re.compile("[\\U0001F300-\\U0001FAFF\\U00002600-\\U000027BF\\U0001F1E6-\\U0001F1FF←-⇿⬀-⯿]")
    _CITATION = re.compile(r"\[来源[：:]\s*[^\]]+\]")

    def __init__(self) -> None:
        self.max_length = 2000
        self.min_length = 10

    def _ends_properly(self, text: str) -> bool:
        tail = text.rstrip()
        return bool(tail) and (tail[-1] in self._ENDING_PUNCT or bool(self._EMOJI.match(tail[-1])))

    def validate(self, text: str) -> tuple[bool, list[str]]:
        warnings: list[str] = []
        if len(text) < self.min_length:
            warnings.append(f"Output too short: {len(text)} chars")
        if len(text) > self.max_length:
            warnings.append(f"Output too long: {len(text)} chars")
        if not self._ends_properly(text):
            warnings.append("Output doesn't end with punctuation")
        if "```" in text:
            warnings.append("Output contains code blocks")
        if self._has_repetition(text):
            warnings.append("Output contains repetition")
        return not warnings, warnings

    def fix(self, text: str) -> tuple[str, int]:
        fixes = 0
        if len(text) > self.max_length:
            truncated = text[: self.max_length]
            last_punct = max(truncated.rfind(mark) for mark in "。！？")
            text = truncated[: last_punct + 1] if last_punct > self.max_length * 0.8 else truncated + "..."
            fixes += 1
        if not self._ends_properly(text):
            text += "。"
            fixes += 1
        if "```" in text:
            text = re.sub(r"```[\s\S]*?```", "", text)
            fixes += 1
        if "\n\n\n" in text:
            text = re.sub(r"\n{3,}", "\n\n", text)
            fixes += 1
        return text, fixes

    def _has_repetition(self, text: str, min_repeat: int = 3) -> bool:
        stripped = re.sub(r"\s+", "", self._CITATION.sub("", text))
        if len(stripped) < 40:
            return False
        if re.search(r"(.{8,})\1{%d,}" % (min_repeat - 1), stripped):
            return True
        seen: dict[str, int] = {}
        for index in range(0, len(stripped) - 15, 5):
            part = stripped[index : index + 15]
            seen[part] = seen.get(part, 0) + 1
            if seen[part] >= 3:
                return True
        return False


class CitationValidator:
    """Validate and extract ``[来源: ...]`` markers."""

    def __init__(self) -> None:
        self.citation_pattern = re.compile(r"\[来源[：:]\s*([^\]]+)\]")

    def validate_citations(self, text: str) -> tuple[bool, list[str]]:
        warnings: list[str] = []
        if not self.citation_pattern.findall(text):
            warnings.append("No citations found in response")
        malformed = re.findall(r"\[来源[^：:][^\]]*\]", text)
        if malformed:
            warnings.append(f"Malformed citations: {malformed}")
        return not warnings, warnings

    def extract_citations(self, text: str) -> list[str]:
        return self.citation_pattern.findall(text)


class ResponsePostProcessor:
    """Apply replacement, masking, repair, and final validation in order."""

    def __init__(self, *, enable_sensitive_filter: bool = True, enable_format_fix: bool = True, enable_citation_check: bool = True) -> None:
        self.sensitive_filter = SensitiveWordFilter() if enable_sensitive_filter else None
        self.regex_replacer = RegexReplacer()
        self.format_validator = FormatValidator()
        self.citation_validator = CitationValidator() if enable_citation_check else None
        self.enable_format_fix = enable_format_fix

    def process(self, text: str) -> PostProcessResult:
        text, replacements = self.regex_replacer.apply(text)
        sensitive = 0
        warnings: list[str] = []
        if self.sensitive_filter:
            text, sensitive = self.sensitive_filter.filter(text)
            if sensitive:
                warnings.append(f"Masked {sensitive} sensitive words")
        _, pre_warnings = self.format_validator.validate(text)
        fixes = 0
        if self.enable_format_fix and pre_warnings:
            text, fixes = self.format_validator.fix(text)
        if self.citation_validator:
            _, citation_warnings = self.citation_validator.validate_citations(text)
            warnings.extend(citation_warnings)
        valid, final_warnings = self.format_validator.validate(text)
        warnings.extend(final_warnings)
        return PostProcessResult(text, replacements, sensitive, fixes, valid, warnings)


def post_process_response(text: str, *, enable_sensitive_filter: bool = True, enable_format_fix: bool = True) -> str:
    """Convenience wrapper for one response."""
    return ResponsePostProcessor(
        enable_sensitive_filter=enable_sensitive_filter,
        enable_format_fix=enable_format_fix,
    ).process(text).text


__all__ = ["CitationValidator", "FormatValidator", "PostProcessResult", "RegexReplacer", "ResponsePostProcessor", "SensitiveWordFilter", "post_process_response"]
