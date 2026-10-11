from rag.llm.vision_model import try_vision_answer
def chat(dialog, messages, stream=True, **kwargs):
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."

    # === B' 视觉旁路：检测到图片问题直接走 GLM-4V ===
    try:
        _last_q = messages[-1].get("content", "")
        if isinstance(_last_q, str) and _last_q:
            _vans = try_vision_answer(_last_q, dialog.tenant_id,
                                       list(dialog.kb_ids) if dialog.kb_ids else [])
            if _vans:
                import logging as _log
                _log.info(f"[vision] bypass answer: {_vans[:80]}")
                yield {"answer": _vans, "reference": [], "prompt": "", "audio_binary": None}
                return
    except Exception as _e:
        import logging as _log
        _log.exception(f"[vision] branch failed: {_e}")
    # === 视觉旁路结束 ===

    # 以下为原有代码，未改动
    if not dialog.kb_ids and not dialog.prompt_config.get("tavily_api_key"):
        for ans in chat_solo(dialog, messages, stream):
            yield ans
        return

    chat_start_ts = timer()
    # ... 后面的代码保持原样