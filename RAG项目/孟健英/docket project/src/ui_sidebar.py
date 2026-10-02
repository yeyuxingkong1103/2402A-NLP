# -*- coding: utf-8 -*-
"""侧边栏：安全横幅、角色选择、会话列表（新建/切换/删除）、系统状态。"""
import streamlit as st  # Streamlit UI 框架

from src import ui_style  # 安全横幅样式

def render(role_mgr, store, bm25_ready: bool, rerank_ready: bool, short_mem=None, long_mem=None) -> str:  # 侧边栏渲染入口，返回当前角色 id；记忆层用于删会话级联清理
    """渲染侧边栏并返回当前角色 id。"""
    with st.sidebar:  # 以下内容全部进侧边栏
        st.subheader("🎭 选择角色")  # 角色选择区标题
        roles = role_mgr.list_roles()  # 全部可选角色
        labels = [f"{r['avatar']} {r['name']} — {r['description']}" for r in roles]  # 单选项文本：头像 + 名称 + 简介
        picked = st.radio("当前角色", labels, index=0, key="role_radio")  # 默认选第一个角色
        role_id = roles[labels.index(picked)]["id"]  # 由选中文案反查角色 id

        ui_style.banner()  # 渲染红色安全横幅；红色安全横幅：固定放在角色选择区下方
        st.divider()  # 分隔线

        from src import image_ocr  # 局部导入：OCR 模块重，用到才加载；图片识别：上传 → OCR（按文件指纹缓存，页面刷新不重跑）→ 拼入后续提问
        if image_ocr.ocr_available():  # 装了 paddleocr 才显示上传区
            st.subheader("📷 图片识别")  # 图片识别区标题
            up = st.file_uploader("上传化验单/药盒/说明书", type=["jpg", "jpeg", "png"], key="ocr_upload")  # 限制常见图片格式
            cache = st.session_state.setdefault("ocr_cache", {})  # 缓存放 session_state，脚本重跑不丢
            if up:  # 有上传文件
                fp = f"{up.name}:{up.size}"  # 文件名 + 大小作指纹，判断是否换了图
                if cache.get("fp") != fp:  # 只有换了文件才重新识别
                    with st.spinner("正在识别图片…"):  # 识别期间转圈提示
                        cache.update(fp=fp, name=up.name, img=up.getvalue(),  # 重新 OCR 并整体更新缓存
                                     text=image_ocr.ocr_image(up.getvalue()))
                st.session_state["ocr_text"] = cache["text"]  # 识别结果供聊天区拼入后续提问
                txt = cache["text"]  # 取文本做预览
                st.caption("识别结果（将拼入后续提问）：" + (txt[:80] + "..." if len(txt) > 80 else (txt or "无")))  # 预览前 80 字，避免刷屏
            else:  # 文件被移除（点 ✕）时
                cache.clear()  # 清缓存
                st.session_state.pop("ocr_text", None)  # 同步清掉识别文本
            if st.session_state.get("ocr_text"):  # 当前有生效的识别结果
                st.info(f"📋 已识别 {len(st.session_state['ocr_text'])} 字，点 ✕ 或清除可取消附图")  # 状态提示：已识别字数
                if st.button("清除图片"):  # 手动清除入口
                    cache.clear()  # 清缓存
                    st.session_state["ocr_upload"] = None  # 重置上传组件本身
                    st.session_state.pop("ocr_text", None)  # 清识别文本
                    st.rerun()  # 重跑刷新界面
            st.divider()  # 分隔线

        st.subheader("💬 会话列表")  # 会话列表区标题
        _sessions(store, role_id, short_mem, long_mem)  # 渲染会话列表（透传记忆层用于删会话级联）

        st.divider()  # 分隔线
        st.subheader("📊 状态")  # 状态区标题
        st.write(f"角色：**{role_mgr.get_role_name(role_id)}**")  # 当前角色名
        st.write(f"BM25 索引：{'✅ 已就绪' if bm25_ready else '⚠️ 未加载'}")  # BM25 索引就绪状态
        st.write(f"Rerank 模型：{'✅ 已启用' if rerank_ready else '⚠️ 回退向量分数'}")  # 重排模型状态，未启用会提示回退
        st.caption("⚠️ 免责声明：本系统仅供学习演示，不能替代医生面诊。")  # 常驻免责声明

    return role_id  # 返回当前角色 id 给主流程

def _sessions(store, role_id: str, short_mem=None, long_mem=None):  # 会话列表渲染（记忆层仅删会话级联用）
    """会话列表：点击切换当前会话，🗑 删除，每行独立上下文。"""
    store.current(role_id)  # 兜底：保证该角色至少有一个会话
    for i, sess in enumerate(store.list(role_id)):  # 遍历该角色所有会话
        col_btn, col_del = st.columns([5, 1])  # 切换按钮宽、删除按钮窄
        mark = "🟢" if i == 0 else "⚪"  # 第一个是当前会话，绿点标记
        label = f"{mark} {sess.get('title')}（{len(sess.get('messages', []))}条）"  # 标题 + 消息数
        if col_btn.button(label, key=f"sw_{i}_{sess.get('session_id')}"):  # 点击切换会话
            store.switch(role_id, i)  # 切到该会话
            st.rerun()  # 重跑刷新
        if col_del.button("🗑", key=f"del_{i}_{sess.get('session_id')}"):  # 点击删除会话
            sid_del = sess.get("session_id")  # 先取出被删会话 id（删除后 sess 脱离列表）
            store.delete(role_id, i)  # 删除 UI 会话
            if short_mem: short_mem.clear(sid_del)  # 级联清 Redis 短期记忆
            if long_mem: long_mem.delete_session("default", role_id, sid_del)  # 级联清 Milvus 该会话长期摘要
            st.rerun()  # 重跑刷新
    if st.button("➕ 新会话"):  # 新建会话入口
        store.new(role_id)  # 建新会话并切换过去
        st.rerun()  # 重跑刷新
