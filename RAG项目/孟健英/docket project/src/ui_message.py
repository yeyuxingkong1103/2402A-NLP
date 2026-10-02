# -*- coding: utf-8 -*-
"""消息渲染：历史气泡、引用展开、反馈收集。"""
import streamlit as st  # Streamlit UI 框架

from src import citation, ui_style  # 引用卡片生成与样式
from src.chat_session import SessionStore  # 会话存储：bad case 落盘要用


def history(msg: dict, role: dict) -> None:  # 历史消息渲染入口
    """渲染一条历史消息（含引用、安全警告、反馈状态）。"""
    assistant = msg.get("role") == "assistant"  # 区分用户/助手气泡
    name = role.get("name", "助手") if assistant else "user"  # 助手用角色名，用户用内置 user 样式
    avatar = role.get("avatar", "") if assistant else None  # 助手带角色头像，用户用默认
    with st.chat_message(name, avatar=avatar):  # 气泡容器
        if assistant:  # 只有助手气泡挂角色徽标
            ui_style.role_chip(role)  # 渲染徽标
        st.write(msg.get("content", ""))  # 消息正文
        if msg.get("image"):  # 附图提问：显示缩略图，OCR 原文折叠备查
            st.image(msg["image"], width=200, caption=f"📷 {msg.get('image_name') or '上传图片'}")  # 缩略图展示上传图片
            if msg.get("ocr_text"):  # 有识别文本才给折叠区
                with st.expander("📄 查看图片识别文本"):  # 默认折叠，避免长文本刷屏
                    st.text(msg["ocr_text"])  # 等宽展示 OCR 原文
        if msg.get("warning"):  # 有安全警告
            st.error(msg.get("warning"))  # 红色重现警告
        if msg.get("citations"):  # 有引用片段
            citations(msg.get("citations"), msg.get("query", ""))  # 渲染引用区
        if msg.get("trace"):  # 有检索追踪信息
            with st.expander("🔍 检索详情"):  # 折叠展示，调试时展开
                st.code(msg.get("trace"))  # 等宽代码块显示追踪
        if msg.get("feedback"):  # 已反馈过的消息
            st.caption(f"已记录反馈：{msg.get('feedback')}")  # 小字回显，防重复点击


def citations(items: list, query: str):  # 引用区渲染：高低相关分组
    """参考资料：≥0.6 直接展示，<0.6 折叠到"更多引用"。"""
    if not items:  # 空列表直接返回
        return  # 无引用不渲染
    high = [it for it in items if it.get("score", 0) >= 0.6]  # >=0.6 高相关直接展示
    low = [it for it in items if it.get("score", 0) < 0.6]  # <0.6 低相关折叠
    if not high:  # 全是低分时
        high = [max(items, key=lambda x: x.get("score", 0))]  # 保底展示最高分那条，引用区不留空
        low = []  # 不再重复折叠
    for i, item in enumerate(high, 1):  # 编号从 1 开始
        with st.expander(citation.label(item, i)):  # 标题含来源与页码
            st.markdown(citation.card(item, query), unsafe_allow_html=True)  # 卡片正文，命中词高亮
    if low:  # 还有低相关条目
        with st.expander(f"📎 更多引用（{len(low)} 条，相关度较低）", expanded=False):  # 默认收起，避免干扰阅读
            for j, item in enumerate(low, len(high) + 1):  # 编号接着高分段续排
                st.markdown(citation.card(item, query), unsafe_allow_html=True)  # 同样的卡片渲染


def feedback(session: SessionStore, record: dict, resp):  # 反馈收集入口
    """👍/👎 收集 bad case，差评样本写入 data/feedback.jsonl。"""
    up, down, _ = st.columns(3)  # 三列布局，只用前两列放按钮
    key = f"fb_{session.get('session_id', '_')}_{len(session.get('messages', []))}"  # 按钮 key 绑定会话与消息序号，全局唯一

    if up.button("👍 有帮助", key=key + "up"):  # 点好评
        record["feedback"] = "👍 有帮助"  # 写回消息记录，刷新后仍显示
        st.rerun()  # 重跑刷新界面

    if down.button("👎 没帮助", key=key + "down"):  # 点差评：收集 bad case
        c = {  # 组装差评样本
            "rating": "bad",
            "session_id": session.get("session_id", "_"),
            "query": record.get("query", ""),
            "content": record.get("content", ""),
            "warning": record.get("warning", ""),
            "citations": [
                {
                    "source": it.get("source"),
                    "page": it.get("page"),
                    "score": round(it.get("score"), 3),
                }
                for it in record.get("citations", [])
            ],
        }
        SessionStore.log_bad_case(c)  # 追加写入 data/feedback.jsonl，供事后复盘
        record["feedback"] = "👎 没帮助"  # 标记已反馈
        st.rerun()  # 重跑刷新