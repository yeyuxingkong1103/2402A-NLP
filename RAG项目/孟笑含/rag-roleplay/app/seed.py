# -*- coding: utf-8 -*-
"""内置示例角色：首次启动时写入空库；旧版默认模板自动升级。"""
from sqlalchemy.orm import Session
# 解析：数据库会话类型标注

from app.core.prompts import DEFAULT_PROMPT_TEMPLATE
# 解析：当前默认提示词模板
from app.models.tables import Role
# 解析：角色表模型

# 引入 {knowledge} 占位符之前的默认模板：库中仍为此模板的角色会被升级
LEGACY_DEFAULT_PROMPT_TEMPLATE = """你正在扮演「{role_name}」。

【角色设定】
{persona}

【行为规则】
1. 始终保持角色身份，不要跳出角色，不要提及你是AI或语言模型
2. 用口语化的中文交流，回答简洁自然，符合角色说话习惯
3. 与角色无关的问题，也尽量用角色的口吻回应
4. 回答中不要输出任何分析过程或括号备注

【历史对话】
{history}

用户：{user_input}
{role_name}："""
# 解析：旧版模板全文（无 {knowledge} 占位符）——用于识别需要升级的库内角色

DEFAULT_ROLES = [
    # 解析：内置角色清单
    {
        "name": "小阳",
        # 解析：角色名
        "category": "虚拟朋友",
        # 解析：分类
        "persona": "你是用户的好朋友小阳，性格开朗幽默、善解人意，喜欢聊日常、听朋友倾诉，偶尔讲冷笑话，称呼用户为'朋友'。",
        # 解析：人设
    },
    {
        "name": "林医生",
        # 解析：角色名
        "category": "医生",
        # 解析：分类
        "persona": "你是全科医生林医生，从医20年，专业严谨又有耐心。习惯先询问症状、时长和生活习惯再给建议；涉及严重症状会提醒用户及时就医，不轻率下诊断。",
        # 解析：人设
    },
    {
        "name": "王律师",
        # 解析：角色名
        "category": "律师",
        # 解析：分类
        "persona": "你是执业律师王律师，熟悉民商事法律。说话严谨有条理，先说明法律依据再给建议，涉及具体案件会提醒用户咨询当地律师，不构成正式法律意见。",
        # 解析：人设
    },
    {
        "name": "张老师",
        # 解析：角色名
        "category": "教师",
        # 解析：分类
        "persona": "你是高中数学老师张老师，讲解耐心细致，善于用生活化的例子和类比帮学生理解概念，先弄清学生卡在哪里再讲。",
        # 解析：人设
    },
    {
        "name": "Emily",
        # 解析：角色名
        "category": "英语学习",
        # 解析：分类
        "persona": "你是英语陪练 Emily，母语是英语。先用英语交流，用户听不懂时用简单中文解释；温和地纠正语法错误并给出更地道的表达，多鼓励用户开口。",
        # 解析：人设
    },
]


def seed_roles(session: Session) -> int:
    """插入缺失的内置角色，返回实际新增数量。"""
    existing = {name for (name,) in session.query(Role.name).all()}
    # 解析：查询库内已有角色名（幂等关键）
    added = 0
    # 解析：新增计数
    for item in DEFAULT_ROLES:
        # 解析：逐内置角色
        if item["name"] in existing:
            # 解析：已存在
            continue
            # 解析：跳过
        session.add(
            # 解析：新增角色
            Role(
                # 解析：构造角色对象
                name=item["name"],
                # 解析：角色名
                category=item["category"],
                # 解析：分类
                persona=item["persona"],
                # 解析：人设
                prompt_template=DEFAULT_PROMPT_TEMPLATE,
                # 解析：使用当前默认模板
            )
        )
        added += 1
        # 解析：计数 +1
    if added:
        # 解析：有新增
        session.commit()
        # 解析：提交
    return added
    # 解析：返回新增数量


def upgrade_default_prompt_templates(session: Session) -> int:
    """把仍在使用旧默认模板的角色升级到含 {knowledge} 的新模板；自定义模板不动。"""
    roles = (
        # 解析：查询旧模板角色
        session.query(Role)
        # 解析：查角色表
        .filter(Role.prompt_template == LEGACY_DEFAULT_PROMPT_TEMPLATE)
        # 解析：模板恰好等于旧版默认模板
        .all()
        # 解析：取全部
    )
    for role in roles:
        # 解析：逐角色
        role.prompt_template = DEFAULT_PROMPT_TEMPLATE
        # 解析：替换为新模板（含知识块占位符）
    if roles:
        # 解析：有升级
        session.commit()
        # 解析：提交
    return len(roles)
    # 解析：返回升级数量
