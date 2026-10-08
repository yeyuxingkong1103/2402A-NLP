"""S10 答案生成的实现模块。

对外的入口有两个：
    · `backend/generate/service.py` —— 服务端（`backend/api/stream.py`）与 CLI 共用
    · `backend/generate_answer.py`  —— 命令行

模块划分（按「什么时机可能出错」切分）：

    prompt    提示词加载与渲染        —— 启动期读取 + 请求期渲染（文件缺失/占位符缺失）
    client    LLM 出网调用 + 重试      —— **本包唯一出网的地方**
    result    生成结果模型 + 引用校验   —— 纯计算（白名单校验、派生字段）
    store     结果落盘                —— **本包唯一碰文件系统的地方**
    service   生成编排                —— 串起上面四个

## 与「逐字话术」的边界

constitution 原则 IV / V 的两条固定话术（紧急前置、免责声明）**不在本包内**，
也**不在提示词内**。

`docs/05` §4.4 规定：拼接逐字话术的位置全仓只有一个 —— `assembly_service.build`
（当前落在 `backend/api/stream.py` 的 `answer_text` 拼装处）。本包产出的只是
**正文**（`GenerationResult.answer_text` 的「已剥离无效标记」的那部分）。

`result.py` 的 `GenerationResult.disclaimer` 会把装配层那份**读出来**放进返回值，
但那是引用，不是第二处定义 —— 全仓检索那两句逐字文本仍然只命中一处。
"""

from pathlib import Path

# backend/generate/__init__.py → parents[0]=generate, [1]=backend, [2]=仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 提示词存放处。
#
# ⚠️ 提示词是**数据**，不是代码 —— 它会被非开发人员审阅与修改（医疗内容需要
#    临床人员把关），因此放在 data/ 下而不是 backend/ 内。放 backend/ 里的
#    后果是可预期的：某次重构顺手改了它，而改的人不知道该找谁确认措辞。
PROMPTS_DIR = PROJECT_ROOT / "data" / "prompts"

ANSWER_PROMPT_FILE = PROMPTS_DIR / "med_qa_answer.md"

# 生成结果落盘处。与 `data/questions/`（S8 的提问留存）对称 ——
# 「问了什么」与「答了什么」放在同一层级、同一命名规则下，查起来才知道去哪找。
ANSWERS_DIR = PROJECT_ROOT / "data" / "answers"

# ---- 回答模式 ----
#
# ⚠️ 两种模式的区别是**表达方式**，不是安全底线。constitution 原则 V 对两者
#    同样生效 —— 个体化诊断/处方/剂量调整在 `clinician` 模式下同样禁止。
#    提示词第八节写明了这一点，这里的常量注释只是同一件事的第二次说明。
MODE_PATIENT = "patient"
MODE_CLINICIAN = "clinician"
MODES: tuple[str, ...] = (MODE_PATIENT, MODE_CLINICIAN)
DEFAULT_MODE = MODE_PATIENT

# ---- 模型与端点默认值 ----
#
# 真实取值一律走 `.env`（`backend/api/config.py` 是唯一声明处），这里只是
# 「.env 没写时用什么」。放在本文件而不是 `client.py`，是因为 `result.py` 落盘时
# 也要记模型名 —— 两边各写一份必然会漂。
#
# ⚠️ `DEFAULT_MODEL` 与 docs/02 §10 记录的 `deepseek-v4-flash` 不一致（本处按
#    调用方给的 `deepseek-flash`）。以 `.env` 的 `LLM_MODEL` 为准 ——
#    这是**文档欠账**，不是代码问题。
DEFAULT_BASE_URL = "https://api.agicto.cn/v1"
DEFAULT_MODEL = "deepseek-flash"

API_KEY_ENV = "AGICTO_API_KEY"

# 重试与预算。见 `client.py` 中关于「总预算 deadline」的说明。
MAX_RETRIES = 2            # 最多重试 2 次 ⇒ 最多 3 次尝试
MIN_RETRY_BUDGET_S = 1.0   # 剩余预算低于此值就不再重试
DEFAULT_TIMEOUT_S = 10.0


class PromptError(Exception):
    """提示词文件缺失、占位符缺失或渲染参数非法。

    ⚠️ 刻意**不继承** `ValueError` —— 与 `backend/query/QueryError`、
    `backend/retrieve/RetrievalError` 同一取向：继承内建异常会让
    `except ValueError` 意外捕获到它，而调用链上游（Pydantic 校验）正在大量抛它。
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
