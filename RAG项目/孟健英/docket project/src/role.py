# -*- coding: utf-8 -*-
"""角色管理：加载人设 + 拼接 system prompt。"""
import json  # 解析 roles.json 角色配置
from pathlib import Path  # 跨平台路径拼接，不怕 Windows/Linux 分隔符差异
from typing import List  # 类型标注

from src.prompts import GUARDRAILS  # 安全约束模板：每个角色的 system prompt 末尾都要拼上它

ROLES_PATH = Path(__file__).resolve().parent.parent / "data" / "roles.json"  # 角色配置文件路径：相对本文件定位，任意目录启动都能找到


class RoleManager:  # 角色管理器：人设加载 + system prompt 组装
    """角色人设加载与 Prompt 构建。"""

    def __init__(self):  # 初始化即加载全部角色到内存
        self._roles = self._load()  # 一次性读盘缓存，后续查询都是 O(1) 字典查找

    def _load(self) -> dict:  # 从 JSON 文件加载角色
        """从 roles.json 加载所有角色。"""
        if not ROLES_PATH.exists():  # 配置文件缺失兜底
            return {}  # 返回空角色表，下游走默认人设
        data = json.loads(ROLES_PATH.read_text(encoding="utf-8"))  # 读全文件解析 JSON，utf-8 保证中文不乱码
        return {r["id"]: r for r in data}  # 列表转 id→角色 的字典：按 id 查询 O(1)

    def list_roles(self) -> List[dict]:  # 取全部角色列表
        """获取角色列表（供 UI 下拉选择）。"""
        return list(self._roles.values())  # 字典值转列表，供 UI 渲染下拉框

    def get(self, role_id: str) -> dict | None:  # 按 id 取角色配置
        """按 ID 获取角色。"""
        return self._roles.get(role_id)  # dict.get：找不到返回 None 而不是抛 KeyError

    def get_role_name(self, role_id: str) -> str:  # 取角色中文名
        """按 ID 获取角色中文名。"""
        role = self._roles.get(role_id)  # 先查角色
        return role["name"] if role else "医疗助手"  # 缺省名兜底，UI 永远有名字可显示

    def build_system_prompt(self, role_id: str) -> str:  # 组装系统指令：人设 + 语气参考 + 禁用术语 + 安全护栏
        """拼接完整 system prompt（人设 + 语气参考 + 角色禁令 + 安全护栏）。"""
        role = self._roles.get(role_id) or {}  # 未知角色按空配置处理，走默认人设
        parts = [role.get("system_prompt", "你是一名医疗助手。")]  # 第一段：人设正文，缺失时用通用医疗助手兜底

        if role.get("catchphrase"):  # 配了口头禅才加语气参考段
            # 口头禅只作语气参考，避免每轮固定开场白
            parts.append(f"【语气参考】偶尔可用口语化说法（例如『{role['catchphrase']}』），但不要每轮都用同一句话开头，更不要用它代替实际内容。")  # 口头禅降权为"偶尔可用"：防止模型每轮固定开场白显得机械

        banned = role.get("banned_terms") or []  # 角色禁用术语表（如心理医生禁用中医术语）
        if banned:  # 有禁用词才加约束段
            parts.append(  # 拼禁用术语约束
                "【禁用术语】回答中不得出现以下任何词汇：" + "、".join(banned) + "。"  # 明确列出禁用词
                + "\n【自检（输出前必须执行）】逐句检查你的回答，如果出现上述任一词汇，"  # 自检指令：让模型输出前逐句自查
                + "立即删除包含该词汇的整个段落，不要保留、不要替换、直接删除。"  # 处理策略写死"整段删除"，防止替换式擦边
            )  # 约束段完成

        parts.append(GUARDRAILS)  # 末尾拼安全护栏："仅根据参考资料回答，没有就说没有"——所有角色共享的防幻觉底线
        return "\n\n".join(parts)  # 空行分段拼接，结构清晰模型更易遵守

    def refresh(self) -> None:  # 热重载角色配置
        """重新加载 roles.json（动态更新角色）。"""
        self._roles = self._load()  # 重读 JSON 替换内存缓存，改配置不用重启服务
