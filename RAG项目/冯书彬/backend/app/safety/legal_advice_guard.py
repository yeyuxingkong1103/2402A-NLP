from backend.app.schemas.chat import ScopeResult


_ALLOWED = ["离婚", "婚姻", "抚养", "财产", "债务", "家暴", "探望", "彩礼", "赡养"]
_UNSUPPORTED = ["合同", "租房", "押金", "劳动", "工伤", "交通事故", "刑事", "借贷", "公司"]


def classify_legal_scope(text: str) -> ScopeResult:
    # 明确非婚姻家事领域时，拒绝个案分析并引导官方渠道。
    if any(keyword in text for keyword in _UNSUPPORTED):
        return ScopeResult(in_scope=False, reason="unsupported_legal_domain")
    # 命中婚姻家事关键词时进入 Task 10 支持范围。
    if any(keyword in text for keyword in _ALLOWED):
        return ScopeResult(in_scope=True, reason="supported_family_law")
    # 不确定领域也不自由分析，避免越界法律建议。
    return ScopeResult(in_scope=False, reason="unsupported_legal_domain")


def out_of_scope_answer() -> str:
    # 固定拒答语不做个案分析，只提供官方求助渠道。
    return "当前仅支持婚姻家事方向的一般法律信息，不能对该问题进行个案分析。建议咨询12348公共法律服务热线、当地法律援助中心、人民法院诉讼服务平台或相关主管部门官方渠道。"
