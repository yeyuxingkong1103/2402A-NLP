"""共享装配工厂：CLI 与 HTTP 共用的重资源装配。

存在的理由（HTTP 接口设计 §三 定调第 3 条）：③b-1 吃过「两处各存一份口径必定
漂移」的亏（Milvus 过滤条件、unit 判定都这么漂过），而重资源的装配比口径更贵 ——
bge-m3 与 reranker 是秒级加载、GB 级内存，两处各装一遍迟早出现「CLI 能跑、服务
不能」。所以装配只写在这里：tools/ask.py 与（后续任务的）app/main.py 都调它。

**工厂本身不缓存**：一次调用 = 一套新对象。缓存是调用方的事 —— HTTP 侧在 lifespan
里建一次存进 app.state（设计 §三 定调第 2 条），CLI 一次问答一个进程、天然只装一次。
做成无状态是**有意**的：模块级单例会让测试之间互相污染（加载过的模型与连过的库
跨用例存活），而「只装一次」这条要求本来就由调用方决定装几次来满足。
"""
from __future__ import annotations

import dataclasses
import logging

# 分层规则的准确形式（设计 §三，2026-09-29 任务 1 复审修订）：core/ 可以 import
# 下层各包，但 db/、retrieval/、generation/、recommend/、ingest/ **不得** import
# core/ —— 只有 main.py 与 api/ 可以（lifespan 要 core.factory、鉴权要 core.security）。
# 说成「谁都不能 import core」会把设计自己的布局判成违规，后续步骤只能在「守规则」
# 与「按设计接线」之间二选一。本文件的 import 全是向下，属允许方向。
# 另记一条本任务新引入的边：recommend/extras.build_extras_fn import
# generation/llm_router 取公众侧 LLM 客户端（费用区块本就要走生成），与既有的
# generation/answer.py → recommend/price_intent 合起来是 recommend ↔ generation
# 双向 —— 两处都有实据（问价意图、公众侧客户端），记录在此；新增跨层边前先读这段
from app.core.config import Config
from app.core.config import load as load_config
from app.db.milvus import get_client
from app.db.mysql import connect, connect_autocommit
from app.generation.answer import Answerer
from app.generation.profiles import SIDE_INTERNAL, SIDE_PUBLIC
from app.ingest.embed import load_model
from app.recommend.extras import build_extras_fn
from app.retrieval.rerank import load_reranker

# 回收失败要留痕。理由与 ③b-1 的留痕同口径：静默吞掉的故障信号会让「连接慢慢泄光」
# 与「一切正常」在日志上完全一样 —— 而那正是本函数要防的那类问题
logger = logging.getLogger(__name__)


def release(conn, client, account_conn=None) -> None:
    """尽力关闭连接、Milvus 客户端与账号连接；单个关闭失败只留 warning，**绝不抛**。

    两个调用方都要「不抛」这个性质，理由各一条：
      - 装配中途失败时（build_services 的 except 支）正在抛的是**真因**（模型目录
        坏了、Milvus 连不上），关闭失败若也抛，就会把真因盖成「close 出错」；
      - lifespan 关闭时抛，会让一次正常的进程退出变成崩溃。
    account_conn 允许缺省（None）：测试里那些只填两三个字段的 Services 不必为了
    释放逻辑补一个新字段，「没建过就不关」本来就是这条循环的语义。
    别的重资源（encoder / reranker）不在这里：torch 模型没有 close，回收交给 GC。
    """
    for name, obj in (("conn", conn), ("account_conn", account_conn),
                      ("client", client)):
        if obj is None:
            continue
        try:
            obj.close()
        except Exception as exc:  # noqa: BLE001 —— 见 docstring：这里不能抛
            logger.warning("回收 %s 失败，已继续回收其余对象：%r", name, exc)


@dataclasses.dataclass(frozen=True)
class Services:
    """一套装好的重资源。字段顺序 = 装配顺序，排查时按它读日志。

    answerer 之外的四个是**同一批对象**（不是各自新建的连接/模型）：法条侧与
    费用侧共用同一个 encoder 与 reranker，重复加载既是浪费显存也让两侧口径可分叉。
    """

    conn: object
    client: object
    encoder: object
    reranker: object
    answerer: Answerer
    # 账号路径专用连接（db/mysql.connect_autocommit 的返回值）。它与 conn 是
    # **两条不同的会话**，这不是浪费：账号读必须看最新提交（停用即时生效），
    # 而业务连接是 autocommit=False 的长事务，读视图停在启动那一刻（实测见
    # connect_autocommit 的 docstring）。放在末尾且无默认值：构造点必须显式
    # 交代它，省略会在运行时变成「登录查不到刚建的账号」这类静默故障
    account_conn: object

    def close(self) -> None:
        """释放这套资源。幂等由调用方保证（lifespan 只调一次）。

        **为什么方法在这里而不是 lifespan 里**（任务 1 交接项）：连接与客户端是
        本对象持有的，谁持有谁释放；写在 lifespan 里就得让 lifespan 知道
        「Services 里哪几个字段是可关闭的」，工厂将来多一个资源，lifespan 会
        静默漏掉它（漏掉的形态是连接泄漏，不是报错）。
        """
        release(self.conn, self.client, self.account_conn)


def build_services(side: str = SIDE_INTERNAL, *, with_extras: bool = False,
                   config: Config | None = None) -> Services:
    """装配一整套重资源：MySQL 连接、Milvus client、bge-m3、精排、Answerer。

    side 只决定一件事：要不要装配附加区块的真依赖（费用检索 + DeepSeek 客户端
    + 留痕）。律师侧**恒不装配**是设计红线（数据不出域），所以「律师侧 + 要区块」
    是调用方的编程错误，当场抛而不是静默忽略：静默忽略会让「配错了」与「配对了」
    在行为上完全一样，而律师侧没有区块本来就是正常的，于是这个错永远暴露不了。
    """
    if with_extras and side != SIDE_PUBLIC:
        raise ValueError(f"只有公众侧装配附加区块（数据不出域）：side={side!r}")
    cfg = load_config() if config is None else config
    # 先验路径再加载：模型目录失效时加载器要等读盘读到一半才报错（bge-m3 是
    # 2.3GB 的全量读），失败点越晚越像链路问题而不像环境问题
    cfg.validate()
    # 逐步记进 built：中途任何一步失败都要把**已建成的**对象按名字回收。
    # 用字典而不是一串局部变量，是因为失败点有五个，逐个写 try 会把回收逻辑
    # 抄五遍；名字留在字典里也让「谁已经建了」不依赖局部变量是否赋值成功
    built: dict[str, object] = {}
    try:
        built["conn"] = connect(**cfg.mysql)
        # 账号路径的第二条连接：登录查行与逐请求查 is_active 都走它（理由与实测
        # 见 db/mysql.connect_autocommit）。建在这里而不是路由里 —— 装配只有
        # 一处（定调第 3 条），路由里现建连接正是任务 3 修过的那个坑
        built["account_conn"] = connect_autocommit(**cfg.mysql)
        built["client"] = get_client(cfg.milvus_uri)
        built["encoder"] = load_model(cfg.embed_model_path)
        # 精排只加载一次并在两条链路间共用：法条侧用它做检索精排（Answerer 持引用），
        # 费用侧用它做同一件事（经 build_extras_fn 传给 fee_search.search）
        built["reranker"] = load_reranker(cfg.reranker_model_path)
        # 律师侧从不构造附加区块（同上面的红线），故真依赖只对公众侧装 —— 副产物是
        # 律师侧不再被「DeepSeek 密钥缺失」牵连，而那本不是它要用的东西
        built["extras_fn"] = (
            build_extras_fn(built["conn"], built["client"], built["encoder"],
                            built["reranker"]) if with_extras else None)
        built["answerer"] = Answerer(
            conn=built["conn"], client=built["client"], encoder=built["encoder"],
            reranker=built["reranker"], extras_fn=built["extras_fn"])
    except BaseException:
        # 本函数失败时不返回任何东西，回收只能在这里做：否则 conn 与 client 当场
        # 泄漏，且在进程外看不见（HTTP 侧的形态是「启动失败、端口没起，但 MySQL 里
        # 多出一个睡眠连接」）。收 BaseException 而不是 Exception：加载模型要几十秒，
        # 这期间按 Ctrl-C 走的正是 KeyboardInterrupt —— 那也会留下连接
        release(built.get("conn"), built.get("client"), built.get("account_conn"))
        raise
    return Services(conn=built["conn"], client=built["client"],
                    encoder=built["encoder"], reranker=built["reranker"],
                    answerer=built["answerer"],
                    account_conn=built["account_conn"])
