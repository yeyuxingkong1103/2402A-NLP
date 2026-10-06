# -*- coding: utf-8 -*-
"""会话存储：按角色维护多会话（新建/切换/删除/自动命名），并记录 bad case。"""
import json  # 差评样本序列化成 JSONL 落盘
import uuid  # 生成全局唯一会话 ID，避免碰撞
from datetime import datetime  # 差评记录打时间戳，便于按时间回溯
from pathlib import Path  # 跨平台路径处理

FEEDBACK_FILE = Path(__file__).resolve().parent.parent / "data" / "feedback.jsonl"  # 差评样本文件：JSONL 一行一条，追加写不用读全文件


class SessionStore:  # 会话存储：按角色分组管理多会话
    """会话状态封装在 st.session_state 里，刷新页面不丢失。"""

    KEY = "sessions_by_role"  # session_state 里的存储键名，结构：{角色id: [会话列表]}

    def __init__(self, state):  # 注入 Streamlit 的 session_state 字典
        self.state = state  # 不自己存数据，全部读写都落到 state 上（页面刷新不丢）

    def _all(self) -> dict:  # 取全量会话字典，缺省自动初始化
        return self.state.setdefault(self.KEY, {})  # setdefault：有则返回、无则建空字典——懒初始化

    def list(self, role_id: str) -> list:  # 取某角色的会话列表
        """某个角色的所有会话（下标 0 为当前会话）。"""
        return self._all().setdefault(role_id, [])  # 同样懒初始化：该角色还没聊过就给空列表

    def current(self, role_id: str) -> dict:  # 取当前会话（约定列表首位即当前）
        """取当前会话，没有就新建一个。"""
        sessions = self.list(role_id)  # 先取该角色的会话列表
        if not sessions:  # 一个会话都没有时
            sessions.append(self._blank())  # 自动补一个空白会话，UI 永远有可显示的内容
        return sessions[0]  # 下标 0 即当前会话

    def new(self, role_id: str) -> dict:  # 新建会话
        """新建会话并置顶为当前会话。"""
        sessions = self.list(role_id)  # 取列表
        session = self._blank()  # 造一个空白会话
        sessions.insert(0, session)  # 插到首位置顶为当前会话
        return session  # 返回新会话供 UI 立即渲染

    def switch(self, role_id: str, index: int) -> None:  # 切换当前会话
        """把第 index 个会话置顶为当前会话。"""
        sessions = self.list(role_id)  # 取列表
        sessions.insert(0, sessions.pop(index))  # 弹出目标会话插到首位：单列表既当存储又当"当前"指针，不用额外字段

    def delete(self, role_id: str, index: int) -> None:  # 删除会话
        """删除会话；删空了自动补一个空会话。"""
        sessions = self.list(role_id)  # 取列表
        sessions.pop(index)  # 按下标删除
        if not sessions:  # 删光了时
            sessions.append(self._blank())  # 自动补空会话，UI 不出现"无会话"死状态

    @staticmethod  # 静态方法：不依赖实例状态
    def _blank() -> dict:  # 构造空白会话结构
        return {  # 会话四要素
            "session_id": f"s-{uuid.uuid4().hex[:8]}",  # 短随机 ID：8 位十六进制本地够用，也充当短期记忆在 Redis 里的 key
            "title": "（新会话）",  # 默认标题，首轮提问后自动改名
            "messages": [],  # 消息列表：UI 渲染与写短期记忆的数据源
            "clarified": False,  # 澄清标记：本会话是否已做过追问澄清，防止反复追问
        }  # 结构结束

    @staticmethod  # 静态方法
    def auto_title(session: dict) -> None:  # 自动命名会话
        """用首轮问题生成会话标题。"""
        first = next((m["content"] for m in session["messages"] if m["role"] == "user"), "")  # 生成器找第一条用户消息；next 带默认值防止空列表报错
        if len(first) > 12:  # 标题过长要截断
            session["title"] = first[:12] + "…"  # 截前 12 字加省略号，侧栏显示整齐
            return  # 截断分支结束
        session["title"] = first or "（新会话）"  # 短标题直接用；空内容兜底默认名

    @staticmethod  # 静态方法
    def log_bad_case(payload: dict) -> None:  # 记录差评样本
        """把差评样本落到 data/feedback.jsonl，用于后续优化。"""
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)  # 确保 data 目录存在，exist_ok 幂等不报错
        payload["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 就地补时间戳字段
        with FEEDBACK_FILE.open("a", encoding="utf-8") as fh:  # 追加模式打开：多条差评累积不覆盖
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")  # JSONL 一行一条：后续逐行读/pandas 都方便，中文不转义
