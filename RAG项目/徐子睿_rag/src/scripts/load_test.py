"""scripts/load_test.py —— 基于 Locust 的对话接口压测脚本。

在链路中的位置：
    独立压测脚本，向已启动的服务（src/api/main.py）发压，不属于运行时链路。

用法：
    python -m pip install locust
    locust -f scripts/load_test.py --host http://127.0.0.1:8902

    然后打开 http://localhost:8089 在 Web 界面设置并发数和加压速率。

压测的目标接口是 /api/v1/chat/completions（非流式），
因为它是全链路最重的一个：一次请求要走完检索 → 精排 → 记忆 → 生成。
"""
from __future__ import annotations

try:
    from locust import HttpUser, between, task
except ImportError:
    # locust 是可选依赖（不在 requirements.txt 里），没装时给出明确提示再抛出 ——
    # 直接让 ImportError 往上冒的话，看到的是"cannot import name HttpUser"，
    # 不容易想到是缺 locust 这个包
    print("locust 未安装：python -m pip install locust")
    raise


class ChatUser(HttpUser):
    """模拟一个真实用户的行为：先注册、建会话，然后反复提问。

    wait_time = between(1, 3)：
        每个虚拟用户在两次任务之间随机等待 1~3 秒。
        这个随机间隔很重要 —— 不加的话所有用户会以固定节奏同时发起请求，
        形成人为的脉冲式压力，测出的数据不能反映真实负载。

    token / session_id 声明为类属性：
        它们会被 on_start 赋值，之后任务方法通过 self 读取。
        在类上先声明一次（而不是只在 on_start 里赋值），
        是为了让读代码的人一眼看到这个类持有哪些状态。
    """

    wait_time = between(1, 3)
    token = ""
    session_id = None

    def on_start(self):
        """每个虚拟用户启动时的前置动作：注册 → 建会话。

        这两步是必需的：对话接口要求携带合法 JWT，
        且必须指定一个自己名下的会话。

        用户名用 f"u{user_count}" 拼出来：
            不同虚拟用户需要不同用户名，否则第二个用户注册时会撞 409。
            用全局并发数做后缀，在同一轮压测里能保证唯一。

        resp.json() 直接取字段而不做错误检查：
            压测环境里接口应该总是正常工作；一旦返回结构不符合预期，
            这里抛出的 KeyError 会被 Locust 记为失败请求，本身也是一种信号。
        """
        username = f"u{self.environment.runner.user_count}"
        resp = self.client.post("/api/v1/auth/register", json={"username": username, "password": "password123"})
        data = resp.json()
        self.token = data["access_token"]
        headers = {"Authorization": f"Bearer {self.token}"}
        session = self.client.post("/api/v1/sessions", headers=headers, json={"role_id": "lawyer"}).json()
        self.session_id = session["session_id"]

    @task
    def chat(self):
        """压测动作：发一轮非流式对话。

        用 stream=False 而不是流式：
            压测关心的是吞吐量和响应时间，非流式一次就能拿到完整结果、
            统计口径清晰。流式请求的"首字节时间"与"总时长"含义不同，
            混在一起统计会让数字难以解释。

        默认走 lawyer 角色、固定问题：
            目的是让每次请求的工作量可比 ——
            问题越长、命中的文档越多，耗时差异越大，
            不固定的话延迟数据的波动来源就不只是服务性能了。
        """
        self.client.post("/api/v1/chat/completions", headers={"Authorization": f"Bearer {self.token}"}, json={"session_id": self.session_id, "message": "合同违约怎么办？", "stream": False})
