# -*- coding: utf-8 -*-
"""全局样式：红色安全横幅、角色气泡配色、知识库/模型补充卡片、引用高亮。"""
import streamlit as st  # Streamlit UI 框架

# 红色安全横幅：常驻在侧边栏角色选择区下方，任何页面都不折叠
BANNER = (
    "<div class='safety-banner'>🚑 本系统不提供诊断和处方，仅供健康科普参考 · 出现胸痛、呼吸困难、意识不清等急症请立即拨打 <b>120</b></div>"  # 横幅 HTML：急症拨打 120 的常驻提醒
)

# 全局 CSS：横幅、气泡圆角、引用块、命中高亮、知识库/角色/引用小标签
_CSS = """
<style>
.safety-banner{background:#D32F2F;color:#fff;padding:8px 12px;border-radius:8px;
 font-weight:600;text-align:center;line-height:1.5;font-size:.78rem;margin:4px 0 10px}
[data-testid="stChatMessage"]{border-radius:12px;padding:8px 10px}
blockquote{background:#FFF8E1;border-left:5px solid #FFB300;padding:10px 14px;
 border-radius:8px;color:#5D4037}
mark{background:#FFEB3B;padding:0 2px;border-radius:3px}
.kb-chip{display:inline-block;background:#E8F5E9;color:#2E7D32;border:1px solid #A5D6A7;
 padding:2px 10px;border-radius:999px;font-size:.8rem;margin-bottom:6px}
.role-chip{display:inline-block;padding:2px 10px;border-radius:999px;color:#fff;
 font-size:.78rem;margin-bottom:6px}
.cite-meta{font-size:.82rem;color:#555;margin-bottom:6px}
.cite-text{line-height:1.75;font-size:.9rem;color:#333}
</style>
"""


# 角色配色规则：用 :has(.r-<id>) 命中气泡内的角色类名，不必碰 Streamlit 内部类名
def apply(roles: list) -> None:  # 全局样式注入入口
    """注入全局 CSS；每个角色按其主色渲染气泡（靠 :has 选择器区分）。"""
    rules = "\n".join(  # 为每个角色生成一条气泡配色规则
        f'[data-testid="stChatMessage"]:has(.r-{r.get("id")})'  # 用 :has 选中含角色标记的气泡，不碰 Streamlit 内部类名
        f'{{background:{r.get("color", "#1565C0")}1A;border-left:4px solid {r.get("color", "#1565C0")}}}'  # 主色加 1A 透明度做底色，左侧 4px 色条
        for r in roles  # 遍历所有角色
    )
    st.markdown(_CSS.replace("</style>", "\n" + (rules or "") + "\n</style>"), unsafe_allow_html=True)  # 把角色规则拼到 </style> 前，一次性注入页面


# 安全横幅：固定渲染在侧边栏角色选择区下方
def banner() -> None:  # 安全横幅渲染入口
    """红色安全横幅（渲染在侧边栏角色选择区下方）。"""
    st.markdown(BANNER, unsafe_allow_html=True)  # 渲染横幅 HTML


def role_chip(role: dict) -> None:  # 角色徽标渲染入口
    """角色徽标（只显示头像+名称，不再挂口头禅，避免每轮重复同一句）。"""
    st.markdown(  # 输出徽标 HTML
        f"<span class='r-{role.get('id', 'other')}'></span>"  # 隐藏 span 挂角色类名，供 :has 选择器命中气泡
        f"<span class='role-chip' style='background:{role.get('color', '#1565C0')}'>"  # 徽标底色用角色主色
        f"{role.get('avatar', '')} {role.get('name', '医疗助手')}</span>",  # 头像 + 名称
        unsafe_allow_html=True,  # 允许渲染原生 HTML
    )