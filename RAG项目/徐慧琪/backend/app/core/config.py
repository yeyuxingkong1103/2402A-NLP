"""配置装载：环境变量 → 值，默认值兜底，失效当场报错。

存在的理由（HTTP 接口设计 §三 定调第 3 条）：CLI 与 HTTP 服务必须共用同一套装配，
而装配的第一件事是「重资源在哪、库连哪里」。这些值原先散在四处 —— 模型路径在
app/ingest/embed.py 与 app/retrieval/rerank.py 的常量里、连接参数在 app/db/*.py 的
默认值里、精排目录在 tools/check_env.py 里又抄了一份字面量。任何一处改了而另一处
没跟上，表现都是「CLI 能跑、服务不能」，且失败点落在模型加载之后。收成一处之后，
改默认值只改这里；默认值本身仍取自各模块的常量，不抄第二份字面量。

边界（有意不做的两件事）：
  ①**只装载、不装配**：建连接、加载模型是 app/core/factory.py 的事。本模块不产生
    任何连着外部服务的对象，所以单测可以随便 import 它而不会加载 2GB 模型；
  ②**不预埋没人用的配置项**：任务 6 起并发上限与限流阈值进本模块（它们有了使用者：
    core/ratelimit.py），端口仍留给部署阶段，JWT 密钥仍归 security.load_secret。
    没有使用者的配置项无法被验证，先加上只会多出一份没人跑的默认值。
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
from collections.abc import Mapping

# 默认值一律取自既有常量（唯一真相源），本模块不复制任何路径或地址字面量
from app.db.milvus import MILVUS_URI
from app.db.mysql import DEFAULT_CONFIG
from app.ingest.embed import DEFAULT_MODEL_PATH
from app.retrieval.rerank import RERANK_MODEL_PATH

# 环境变量名统一加 FL_ 前缀：本机用户环境变量里已有别的项目留下的 MYSQL_* 一类
# 名字，不加前缀会与它们撞车，而撞车的后果是连到别人的库且不报错。
# 唯一不在此列的既有变量是 DeepSeek 密钥：那个名字（api）由用户 2026-09-23 指定，
# 且归 llm_router 管（DEEPSEEK_KEY_ENV 是它的唯一真相源，本模块不重复定义）
ENV_MILVUS_URI = "FL_MILVUS_URI"
ENV_EMBED_MODEL_PATH = "FL_EMBED_MODEL_PATH"
ENV_RERANK_MODEL_PATH = "FL_RERANK_MODEL_PATH"

# MySQL 五个字段各自可覆盖：mysql.DEFAULT_CONFIG 的注释已声明「生产按技术方案
# 附录 B 走 .env，真实值不入仓库」，而逐个字段比解析 DSN 少一层易错的字符串处理
ENV_MYSQL = {
    "host": "FL_MYSQL_HOST",
    "port": "FL_MYSQL_PORT",
    "user": "FL_MYSQL_USER",
    "password": "FL_MYSQL_PASSWORD",
    "database": "FL_MYSQL_DATABASE",
}

# 模型目录里必须真在的那个文件。查它们是因为缺了**不报错、只安静地退化**：
# bge-m3 缺 sparse_linear.pt 时 FlagEmbedding 会随机初始化 sparse 头（实测同一
# 文本两次加载得到 [17,17] 与 [10,10]），混合检索的稀疏路当场变成噪声；精排缺
# model.safetensors 则连加载都过不去。两者的报错点都在加载之后，故提前到装配之前
REQUIRED_MODEL_FILES = {
    "embed_model_path": "sparse_linear.pt",
    "reranker_model_path": "model.safetensors",
}


class ConfigError(RuntimeError):
    """配置失效。启动时当场抛：让「环境不对」与「链路不对」在报错上可分。"""


class MissingRateSaltError(ConfigError):
    """限流加盐值缺失或过短（任务 6）。

    与 security.MissingSecretError 同族：**配置故障不许伪装成业务拒绝**（任务 2 的
    裁决）。它是 ConfigError 的兄弟而不是 ApiError 的子类，所以永远映射不到
    404/429/400 —— 它只会落进 500 那条兜底（通用文案，变量名只进日志）。三条
    候选里选它的理由见 ratelimit.client_key 的注释：fail-open（悄悄不限流）＝
    本项目最怕的静默失效，fail-closed 成 429 ＝ 拿一条假业务理由骗调用方。
    """


@dataclasses.dataclass(frozen=True)
class Config:
    """装配需要的全部「在哪」。frozen：装配期间改它没有任何合法用途。"""

    milvus_uri: str
    mysql: dict
    embed_model_path: str
    reranker_model_path: str

    def validate(self) -> None:
        """检查两个模型目录里该在的文件真在；缺则抛 ConfigError，一次列全。

        为什么不做「连不上就报错」式的全量自检：活体检查（Milvus / MySQL /
        Ollama）要发网络请求，开销与失败语义都与本函数不同，那是
        tools/check_env.py 的职责。这里只回答「这次装配会不会因为路径不对而白跑」。
        """
        missing = [str(pathlib.Path(getattr(self, field)) / filename)
                   for field, filename in REQUIRED_MODEL_FILES.items()
                   if not (pathlib.Path(getattr(self, field)) / filename).exists()]
        if missing:
            # 一次列全而不是遇错即抛：环境缺件常是成片的（换了机器/挂了盘），
            # 让人一次看全比重跑几次各看一条便宜
            raise ConfigError("模型文件缺失，装配前自检未通过：" + "、".join(missing))


def load(env: Mapping[str, str] | None = None) -> Config:
    """从环境变量装载配置。env 可注入（默认 os.environ）——单测不必改进程环境。"""
    env = os.environ if env is None else env
    return Config(milvus_uri=_text(env, ENV_MILVUS_URI, MILVUS_URI),
                  mysql=_mysql_config(env),
                  embed_model_path=_text(env, ENV_EMBED_MODEL_PATH, DEFAULT_MODEL_PATH),
                  reranker_model_path=_text(env, ENV_RERANK_MODEL_PATH,
                                            RERANK_MODEL_PATH))


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    """取一个字符串配置；未设置与空串都算未设置，退回默认值。

    空串当未设置是既有口径（llm_router 对密钥的判法就是这个），这里必须与它一致：
    .env 里 `FL_MILVUS_URI=` 这样的空行不该把服务指向一个空地址 —— 那会连到一个
    看起来像配置好了、实际连不上的地方。
    """
    return (env.get(name) or "").strip() or default


# ---- 应用层防滥用（任务 6，设计 §六）-----------------------------------------
# 这里只放**阈值**，限流本身的实现（滑动窗口、并发闸门、中间件）在 core/ratelimit.py。
# 阈值走环境变量的理由与模型路径同一件事：调参不该改代码（改代码要重跑全量闸门），
# 而默认值必须写在代码里一处、能被测试读到 —— 调参的人不该去猜「不设时是多少」。
ENV_PUBLIC_CONCURRENCY = "FL_PUBLIC_CONCURRENCY"
ENV_RATE_LIMIT_MAX = "FL_RATE_LIMIT_MAX"
ENV_RATE_LIMIT_WINDOW_S = "FL_RATE_LIMIT_WINDOW_S"
ENV_RATE_LIMIT_MAX_KEYS = "FL_RATE_LIMIT_MAX_KEYS"
ENV_RATE_LIMIT_SALT = "FL_RATE_LIMIT_SALT"

# 公众侧并发上限 = 20：设计 §六 明写（对齐 AC-14），不是本模块自定的
DEFAULT_PUBLIC_CONCURRENCY = 20
# 60 秒窗口内 30 次/IP。这个数字的算账（写进任务 6 报告）：一次公众问答要数秒，
# 正常用户 1 分钟最多十几次；30 给了 2~3 倍余量，而脚本 1 秒就能打满 30 发，
# 于是持续滥用被压到 ≤30 次/分钟/IP。窗口取 60s 是因为它要覆盖「一个人连着问
# 七八个问题」的自然节奏，比它短会让正常用户撞线，比它长则脚本的持续速率不变
DEFAULT_RATE_LIMIT_MAX = 30
DEFAULT_RATE_LIMIT_WINDOW_S = 60.0
# 上限键数：匿名端点人人可造键（每源 IP 一个），不设上限＝一条内存耗尽通道。
# 一万个键 × 每条几十字节 ≈ 亚 MB 量级，而单机部署一分钟内见到一万个不同源 IP
# 已经是分布式压测的形态了（再多也只是把计数精度让出去，见 ratelimit 的淘汰策略）
DEFAULT_RATE_LIMIT_MAX_KEYS = 10_000
# 加盐值的长度下限。与 security.MIN_SECRET_LEN（32）不同：那个是签名密钥，破了
# 就能伪造 token；这个盐只用来挡住「拿 IPv4 全空间（2^32）做彩虹表反推 IP」，
# 16 字符（≈128 bit）已远超枚举成本，再长只是让运维更难凑
MIN_RATE_SALT_LEN = 16


@dataclasses.dataclass(frozen=True)
class RateLimitSettings:
    """限流与并发上限的四个阈值（frozen：运行期改它没有合法用途）。"""

    limit: int
    window_s: float
    max_keys: int
    concurrency: int


def load_rate_limit(env: Mapping[str, str] | None = None) -> RateLimitSettings:
    """从环境变量装载四个阈值。env 可注入（单测不必改进程环境）。

    与模型路径不同，这里**填了非数字就当场抛**（见 _number）而不退回默认值：
    阈值退回默认值的形态是「运维以为限了 5 次，实际是 30 次」，没有任何红灯。
    """
    env = os.environ if env is None else env
    return RateLimitSettings(
        limit=_number(env, ENV_RATE_LIMIT_MAX, DEFAULT_RATE_LIMIT_MAX, int),
        window_s=_number(env, ENV_RATE_LIMIT_WINDOW_S,
                         DEFAULT_RATE_LIMIT_WINDOW_S, float),
        max_keys=_number(env, ENV_RATE_LIMIT_MAX_KEYS,
                         DEFAULT_RATE_LIMIT_MAX_KEYS, int),
        concurrency=_number(env, ENV_PUBLIC_CONCURRENCY,
                            DEFAULT_PUBLIC_CONCURRENCY, int))


def _number(env: Mapping[str, str], name: str, default, cast):
    """取一个数值配置；未设置与空串算未设置（与 _text 同口径），非法值当场抛。"""
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 不是数值：{raw!r}") from exc


def load_rate_salt(env: Mapping[str, str] | None = None) -> str:
    """取限流的加盐值；缺失或过短当场抛 MissingRateSaltError。

    **没有默认盐**（设计 §六 的「不存 IP 原文」只有配了盐才成立）：一个写死的默认盐
    等于把 IP 哈希重新变成可反推的（2^32 的枚举量对一台笔记本是几分钟的事），而
    服务带着公开盐一切正常地跑着，比 500 糟得多 —— 与 security.load_secret 同一条
    理由。报错只提变量名与长度，不回显取值。
    """
    raw = ((os.environ if env is None else env).get(ENV_RATE_LIMIT_SALT) or "").strip()
    if not raw:
        raise MissingRateSaltError(
            f"环境变量 {ENV_RATE_LIMIT_SALT} 未设置或为空；限流加盐值必须由环境提供")
    if len(raw) < MIN_RATE_SALT_LEN:
        raise MissingRateSaltError(
            f"环境变量 {ENV_RATE_LIMIT_SALT} 过短（{len(raw)} 字符 < "
            f"{MIN_RATE_SALT_LEN}）：盐太短时 IP 哈希可被枚举反推")
    return raw


def _mysql_config(env: Mapping[str, str]) -> dict:
    """连接参数 = mysql.DEFAULT_CONFIG 打底 + 环境变量覆盖。

    port 当场转 int：pymysql 内部会自己 int()，但那时抛的 ValueError 不带字段名，
    报错点落在驱动深处，而「填了非数字」这种错误越早说越省事。
    """
    config = dict(DEFAULT_CONFIG)
    for key, name in ENV_MYSQL.items():
        raw = (env.get(name) or "").strip()
        if not raw:
            continue
        if key == "port":
            try:
                config[key] = int(raw)
            except ValueError as exc:
                raise ConfigError(f"环境变量 {name} 不是整数：{raw!r}") from exc
        else:
            config[key] = raw
    return config
