# -*- coding: utf-8 -*-
"""对话区：处理输入 → 检索决策 → 分模式渲染（流式/澄清/急诊/无内容）。"""
import streamlit as st  # Streamlit UI 框架

from src import safety, ui_message, ui_style  # 安全层/消息渲染/样式
from src.chat_session import SessionStore  # 会话存储：自动生成会话标题要用
from src.intent import KNOWLEDGE  # 意图常量
from src.rag_chain import CAPABILITY, CLARIFY, EMERGENCY, NO_CONTENT  # 四种非流式应答模式，决定渲染分支


def render(role_mgr, session, role_id, chain, embedder, bm25, short_mem, long_mem):  # 聊天区主入口：每轮重跑都先重绘历史再处理新输入
    """渲染历史消息并处理本轮输入。"""
    role = role_mgr.get(role_id)  # 当前角色配置（名字/头像/配色）
    for msg in session.get("messages", []):  # 先重绘全部历史消息
        ui_message.history(msg, role)  # 单条气泡渲染（含引用/警告/反馈状态）

    typed = st.chat_input("请描述您的症状或问题...")  # 底部输入框，回车即触发脚本重跑
    pending = session.pop("pending", None)  # 澄清选项被点选后从 pending 接续
    if pending:  # 上一轮点了澄清选项
        _ask(role_mgr, session, role_id, pending, chain, embedder, bm25, short_mem, long_mem)  # 选项文本直接作为本轮提问
    else:  # 自由输入分支
        session.pop("clarified", None)  # 自由输入：清掉上一轮的澄清标记
        if typed:  # 有输入才处理
            _ask(role_mgr, session, role_id, typed, chain, embedder, bm25, short_mem, long_mem)  # 正常走一轮问答


def _ask(role_mgr, session, role_id, prompt, chain, embedder, bm25, short_mem, long_mem) -> dict:  # 一轮问答核心：全流程异常兜住
    """一轮问答：存档 → 检索决策 → 渲染 → 写入短期记忆。所有异常兜住。"""
    import traceback  # 仅异常路径用，局部导入
    # OCR 文本拼入检索 query（侧栏上传图片识别后缓存；用户气泡只显示图片，不显示原文）
    oc = st.session_state.get("ocr_cache", {})  # 取侧边栏缓存的图片与识别结果
    ocr = st.session_state.get("ocr_text", "")  # 识别出的文本
    query = f"[图片识别内容：{ocr[:1000]}] {prompt}" if ocr else prompt  # 有图就把 OCR 文本拼进检索 query，截断防爆长度

    role = role_mgr.get(role_id)  # 角色配置
    sid = session.get("session_id")  # 会话 ID：写短期记忆与按钮 key 都要用
    history = list(session.get("messages", []))  # 拷一份历史，供多轮改写参考

    # 本轮提问先入档，保证气泡重绘与短期记忆一致
    extra = {"image": oc["img"], "image_name": oc.get("name", ""), "ocr_text": ocr[:500]} if oc.get("img") else {}  # 有图才把图片与 OCR 摘要一并入档
    session["messages"].append({"role": "user", "content": prompt, **extra})  # 用户消息入档
    ui_message.history(session["messages"][-1], role)  # 立刻渲染刚发的用户气泡
    if len(session["messages"]) == 1:  # 本会话首条消息
        SessionStore.auto_title(session)  # 用首问自动生成会话标题

    with st.chat_message(role.get("name", "助手"), avatar=role.get("avatar", "")):  # 助手气泡容器：名字与头像跟角色走
        ui_style.role_chip(role)  # 角色徽标
        try:  # 检索与生成的所有异常在此兜住
            with st.spinner("正在检索知识库…"):  # 检索期间转圈提示
                resp = chain.prepare(  # 检索决策全链路，返回五种应答模式之一
                    role_id, query, embedder, bm25, short_mem, long_mem,
                    sid, session.get("clarified", False), history,
                )
            record = _render(role_mgr, session, role_id, prompt, resp, chain)  # 按模式渲染并把助手消息落档
        except Exception as exc:  # 任何异常都不让对话中断
            traceback.print_exc()  # 堆栈打到控制台便于定位
            st.error(f"❌ 出错了：{exc}")  # 界面上给用户看友好错误
            record = {  # 构造错误占位消息
                "role": "assistant",
                "content": f"⚠️ 系统错误：{exc}\n\n请重新提问，或稍后再试。",
                "warning": "", "query": prompt, "citations": [], "trace": "",
            }
            session["messages"].append(record)  # 错误消息也入档，刷新后仍可见

    # 写入短期记忆（Redis 不可用时 short_mem 为 None，静默降级）
    if record and short_mem:  # 两个条件：本轮出了记录 且 Redis 连接正常
        short_mem.add(sid, "user", query)  # 用户侧：存含 OCR 拼接后的完整 query
        short_mem.add(sid, "assistant", record["content"])  # 助手侧：存回答正文
    return record  # 返回本轮记录


def _render(role_mgr, session, role_id, prompt, resp, chain):  # 按应答模式分四路渲染
    """按应答模式渲染，并把助手消息写回会话。"""
    role = role_mgr.get(role_id)  # 角色配置

    # ① 急症/系统能力：直接展示预设文案，跳过检索
    if resp.mode in (EMERGENCY, CAPABILITY):  # 急症/能力自述：预设文案直出，零延迟零幻觉
        st.error(resp.message) if resp.mode == EMERGENCY else st.info(resp.message)  # 急症红色告警，能力问题蓝色提示
        record = _store(session, resp, resp.message)  # 入档
        ui_message.feedback(session, record, resp)  # 渲染 👍/👎 反馈按钮
        return record  # 返回记录

    # ② 信息不足：给选项按钮，点选后写 pending/clarified 再重跑
    if resp.mode == CLARIFY:  # 信息不足：给可点击的补全选项
        st.write(resp.message)  # 先显示追问话术
        record = _store(session, resp, resp.message)  # 追问先落档再画按钮
        cols = st.columns(len(resp.options))  # 每个选项一列
        for i, option in enumerate(resp.options):  # 逐个渲染选项按钮
            key = f"opt_{session.get('session_id', '_')}_{len(session.get('messages', []))}_{i}"  # key 含会话与消息序号，防重跑时组件冲突
            if cols[i].button(option, key=key):  # 用户点选了某个选项
                session["clarified"] = True  # 标记已澄清，下轮不再二次追问
                session["pending"] = option  # 选项文本暂存，重跑后接续为提问
                st.rerun()  # 立即重跑进入下一轮问答
        return record  # 返回记录

    # ③ 无命中：黄色警示 + 按意图类型给兜底回答（情绪/症状/通用）
    if resp.mode == NO_CONTENT:  # 知识库无命中：警示 + LLM 通用知识兜底
        st.warning(resp.message)  # 黄色提示非知识库内容
        tips = chain.general_tips(role_id, resp.query or prompt, resp.intent or "")  # 按意图生成兜底建议（情绪/症状/通用分路）
        st.markdown(f"<blockquote> <b>补充提示</b><br>{tips}</blockquote>", unsafe_allow_html=True)  # 引用块样式展示兜底文案
        answer = f"\n\n> 💡 补充提示\n> {tips}"  # 同步转成 markdown 引用格式存入历史
        record = _store(session, resp, answer)  # 入档
        ui_message.feedback(session, record, resp)  # 反馈按钮
        return record  # 返回记录

    # ④ 正常作答：意图说明 → 知识库标记 → 流式输出 → 双重安全校验
    labels = {"knowledge": ("知识科普", "直接作答"), "lab_report": ("化验单解读", "直接解读")}  # 意图到中文标签的映射
    type_label, tip = labels.get(resp.intent, ("个人病情", "先结论后追问"))  # 默认按个人病情模式处理
    st.caption(f"🧭 {type_label}模式 · {tip}")  # 顶部小字标注当前模式
    if resp.citations:  # 有引用才标基于知识库
        st.markdown("<span class='kb-chip'>📗 以下内容基于知识库指南</span>", unsafe_allow_html=True)  # 绿色知识库徽标
        # P2: 相关度有限时前置提示，避免与末尾详细回答冲突
        if max(c.get("score", 0) for c in resp.citations) < 0.5:  # 最高分不足 0.5 视为相关度有限
            st.info("📋 本轮资料相关度有限，回答仅供参考，建议咨询医生。")  # 前置免责提示，避免与末尾回答冲突
    answer = st.write_stream(chain.stream_answer(role_id, resp.prompt))  # 流式输出：边生成边显示，降低首字延迟
    if not answer or not answer.strip():  # 流式返回空时的兜底
        answer = "⚠️ 未能生成回答，请重新提问或换个说法。"  # 占位文案，绝不让用户看到空白
        st.write(answer)  # 渲染占位文案
    drugs = safety.unverified_drugs(answer, "".join(c["text"] for c in resp.citations))  # 答案里的药名若不在引用中，视为未经验证
    warning = safety.drug_warning(drugs) if drugs else ""  # 生成用药警示语
    off_role = safety.banned_hits(answer, role.get("banned_terms", []))  # 检查是否出现角色禁用术语
    if off_role:  # 命中禁用术语则追加提醒
        warning += (  # 拼接角色一致性提醒文案
            "⚠️ 角色一致性提醒：本轮出现了不属于"
            f"{role.get('name', '本角色')}"
            "的术语「" + "、".join(off_role) + "」，请以正文中的实际建议为准。"
        )
    if warning:  # 有任何警示
        st.error(warning)  # 红色展示警示

    record = _store(session, resp, answer, warning)  # 正常答案入档（含引用与追踪）
    ui_message.feedback(session, record, resp)  # 反馈按钮
    return record  # 返回记录


def _store(session, resp, content: str, warning: str = ""):  # 助手消息落档小工具
    """助手消息入档（引用与检索详情一并保存，供历史重绘）。"""
    record = {  # 组装消息记录：引用与检索详情一并保存
        "role": "assistant",
        "content": content,
        "warning": warning,
        "query": resp.query,
        "citations": resp.citations,
        "trace": resp.trace,
    }
    session["messages"].append(record)  # 写入会话历史
    return record  # 返回记录供写短期记忆