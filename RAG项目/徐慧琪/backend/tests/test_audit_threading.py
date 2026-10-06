# 审计写**不在事件循环上**的判据（终审 I-1 的修复轮）。
#
# 为什么单独一个文件：test_audit.py 已顶在 300 非空行闸门，而它其余判据全要真连库
# （这里只需要轻替身）；test_main_ratelimit.py 的主题是限流接线。本文件判的是
# **中间件把阻塞写放在哪条线程上**，与「写没写对」是两件事。
#
# 计时不写脆：慢写替身等一个事件（带超时，防「挂死而不是变红」），用例在写**还
# 压着**时打并发 /healthz，然后才放行；判据是耗时上界，余量三个数量级。
import threading
import time

from fastapi.testclient import TestClient

from app import main
from app.generation.answer import QAResult
from tests._fakes_http import FakeServices, fake_factory

URL = "/api/v1/public/qa"
BODY = {"question": "房东不退押金怎么办"}


class _OkAnswerer:
    """假 Answerer：直接回 ok（本文件判线程，业务只需一条干净的 200）。"""

    def answer(self, question: str, side: str) -> QAResult:
        return QAResult(status="ok", answer="依据《民法典》第七百三十三条……")
# 慢写替身最多按住多久：写跑在循环上时 /healthz 会被卡住这么久（红在耗时上界，
# 而不是把用例挂死）
HOLD_S = 5.0
# /healthz 的松上界：正常实现是亚毫秒级（假连接、假 Milvus），留三个数量级余量
BOUND_S = 1.0


class _SlowWriter:
    """审计写替身：进来先自报（entered），再等放行（release，最多 HOLD_S）。

    超时是硬要求：event 环被按住时 release 要等 /healthz 返回后才置，没有超时
    整个用例会挂在循环里 —— 那不是「改坏→变红」，是「改坏→没信号」。
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, conn, fields) -> None:
        self.entered.set()
        self.release.wait(timeout=HOLD_S)


def test_a_slow_audit_write_does_not_block_the_event_loop():
    """慢写进行中时，并发 /healthz 必须立刻得到响应（终审 I-1 的判据）。

    终审探针实测：写跑在事件循环上时，慢写 1.0s 会把并发的 /healthz 拖到 0.853s
    （控制组 0.001s）——「失败不阻断」之外，慢也会阻断**全服务**。前置条件是
    「写仍压着」（entered 已置、release 未置）：没有这个前置，「写完之后 healthz
    有多快」对同步实现也恒绿。
    """
    slow = _SlowWriter()
    factory_fn, _ = fake_factory(FakeServices(answerer=_OkAnswerer()))
    app = main.create_app(services_factory=factory_fn)
    app.state.audit_writer = slow
    with TestClient(app, raise_server_exceptions=False) as client:
        posted: list = []
        thread = threading.Thread(
            target=lambda: posted.append(client.post(URL, json=BODY)))
        thread.start()
        assert slow.entered.wait(5), "审计写没有被调到（替身没接上？）"
        started = time.monotonic()
        health = client.get("/healthz")
        elapsed = time.monotonic() - started
        assert not slow.release.is_set(), "/healthz 返回时慢写已结束，前置没压住"
        slow.release.set()
        thread.join(10)
        assert not thread.is_alive(), "发了慢写的那一发没有返回"
    assert [r.status_code for r in posted] == [200], "问答没成功，慢写判据不成立"
    assert health.status_code == 200
    assert elapsed < BOUND_S, (
        f"审计写还压着时 /healthz 花了 {elapsed:.3f}s（应 < {BOUND_S}）："
        "阻塞写又跑回事件循环上了")
