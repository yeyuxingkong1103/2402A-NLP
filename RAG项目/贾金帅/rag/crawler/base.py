# -*- coding: utf-8 -*-
"""
爬虫基础设施（全模块唯一入口）：浏览器驱动 + HTTP 工具，禁止别处再 import DrissionPage。

为什么用浏览器：
  NMPA 全站有「瑞数」动态令牌反爬，纯 requests 一律返回 412。这里用 DrissionPage
  驱动本机 Edge 完成瑞数挑战（无头模式会被检测，故最小化真实窗口），再借页面内的
  pajax 对象（官方前端自带、自动带 sign/时间戳/令牌）直接调官方接口，与真人行为一致。

  chictr / wanfang 站点前置 WAF 或 SPA，同样用浏览器自然通过，不逆向接口。

设计：所有爬虫共享这一份浏览器与 HTTP，调用方通过 self.page（ChromiumPage）做站点
专属的 DOM 抓取；NMPA 的 pajax 接口由 api_query / api_detail 统一封装。
"""
import json
import logging
import os
import ssl
import sys
import time
import urllib.request

# ======================= 日志（全模块唯一出口） =======================
# 为什么不用 print：爬虫一跑就是几小时，print 没有时间戳、没有级别、留不下文件，
# 出问题根本没法回溯。这里统一配置， spiders.py / cli.py 一律用 get_logger() 取。
#
# 日志文件路径由 cli.py 从 schema.PATHS 传入（路径铁律仍归 schema 管）；
# 直接 import spiders 而不过 cli 时，退化为「只输出到控制台」。
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"


def setup_logging(log_file=None, level=logging.INFO):
    """初始化 crawler 日志：控制台 + 可选文件。重复调用安全（只初始化一次）。"""
    logger = logging.getLogger("crawler")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(LOG_FORMAT, LOG_DATEFMT)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        try:
            d = os.path.dirname(log_file)
            if d:
                os.makedirs(d, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception as e:  # noqa: BLE001
            logger.warning("日志文件不可用（%s），仅输出到控制台：%s", log_file, e)
    return logger


def get_logger():
    """全模块统一取 logger 的入口。未初始化时自动退化为控制台输出。"""
    return setup_logging()


# ======================= 浏览器配置（唯一出处） =======================
# 默认 Edge 安装路径；换浏览器/机器时不用改代码，设环境变量 EDGE_PATH 即可覆盖。
EDGE_PATH = os.environ.get(
    "EDGE_PATH",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)
HOME_URL = "https://www.nmpa.gov.cn/datasearch/home-index.html"

# 接口轮询参数
POLL_INTERVAL = 2   # 秒：每次查询后轮询间隔
POLL_TIMEOUT = 15   # 次：最多轮询 15 次（= 30 秒）

# queryDetail 滑动窗口节流：约 10 次成功 / 105s，窗口内请求立即报错且不计入配额。
# 重试必须等满 backoff（> 窗口长），否则重试仍落在窗口内必败。
DETAIL_RETRIES = 2
DETAIL_BACKOFF = 120


# ======================= HTTP：GET 带重试 =======================
try:
    import requests
except ImportError:
    requests = None

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")


class Resp:
    """统一响应对象：status / text / content"""
    __slots__ = ("status", "text", "content")

    def __init__(self, status, text, content):
        self.status, self.text, self.content = status, text, content


def _get_unverified(url, timeout, headers):
    """SSL 证书异常时的降级：urllib 不校验证书。"""
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        body = r.read()
        return Resp(r.status, body.decode("utf-8", "replace"), body)


def http_get(url, timeout=30, retries=2, backoff=2.0, headers=None):
    """GET 带重试；SSL 证书异常自动降级为不校验证书；全失败返回 None。"""
    hdrs = {"User-Agent": UA}
    if headers:
        hdrs.update(headers)
    last = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(backoff * attempt)
        try:
            if requests is None:
                return _get_unverified(url, timeout, hdrs)
            r = requests.get(url, timeout=timeout, headers=hdrs)
            r.raise_for_status()
            return Resp(r.status_code, r.text, r.content)
        except Exception as e:  # noqa: BLE001
            if requests is not None and isinstance(e, requests.exceptions.SSLError):
                try:
                    return _get_unverified(url, timeout, hdrs)
                except Exception as e2:  # noqa: BLE001
                    last = e2
            else:
                last = e
    get_logger().warning("GET 失败: %s -> %r", url, last)
    return None


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ======================= 浏览器驱动 =======================
class BrowserSpider:
    """浏览器驱动基类：启动 / 关闭 / NMPA 接口 / 通用导航。

    子类或调用方通过 self.page 做站点专属 DOM 抓取（如 chictr 的表格、wanfang 的列表）。
    """

    def __init__(self, interval=3, max_pages=10):
        self.interval = interval      # 相邻请求最小间隔（秒）
        self.max_pages = max_pages    # 列表接口单次最多翻页数
        self.page = None
        self._last = 0.0

    # ---------- 生命周期 ----------
    def start(self):
        from DrissionPage import ChromiumPage, ChromiumOptions
        # 瑞数检测 headless：必须伪装成普通 Chrome UA 才放行（否则返回空白页）。
        # 沙箱/无 GUI 环境只能 headless；有桌面的本机同样可用本配置（更稳定）。
        UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
        co = (ChromiumOptions()
              .headless(True)
              .set_browser_path(EDGE_PATH)
              .set_argument("--disable-blink-features=AutomationControlled")
              .set_argument("--no-sandbox")
              .set_argument("--disable-gpu")
              .set_argument("--disable-dev-shm-usage")
              .set_argument("--user-agent=" + UA))
        self.page = ChromiumPage(co)
        return self

    def close(self):
        try:
            if self.page:
                self.page.quit()
        except Exception:
            pass

    def restart(self):
        """关闭并重新拉起整个浏览器进程（会话彻底失效时兜底）。

        瑞数按「浏览器进程 + IP」发会话令牌，进程内的 page.get 重载在某些无头环境
        下不能稳定重过挑战；而全新浏览器进程（全新临时 profile）等价于一次全新启动，
        实测一定能重过瑞数拿到新令牌。所以会话彻底失效时，直接重启进程最稳。
        """
        logger = get_logger()
        logger.info("重启浏览器进程以重建会话...")
        try:
            if self.page:
                self.page.quit()
        except Exception:  # noqa: BLE001
            pass
        self.page = None
        self.start()
        return self

    def _refresh_session(self, retries=3):
        """瑞数会话过期自愈：先重载首页（重新过挑战拿新令牌，最快），
        重载失败再重启整个浏览器（最稳）。成功返回 True，全失败返回 False。

        NMPA 的 pajax 接口依赖首页会话令牌，令牌有 TTL，跑着跑着会静默失效——
        表现是 api_query 开始返回 None。这里自动恢复，无需人工干预。
        """
        logger = get_logger()
        for i in range(retries):
            try:
                self.load_home()
                logger.info("会话已重建（重载首页，第 %d 次）", i + 1)
                return True
            except Exception as e:  # noqa: BLE001
                logger.warning("首页重载失败（%s），尝试重启浏览器兜底", e)
                try:
                    self.restart()
                except Exception as e2:  # noqa: BLE001
                    logger.warning("浏览器重启失败：%s", e2)
            time.sleep(2)
        return False

    def load_home(self, retries=4):
        """打开 NMPA 首页并等 Vue 应用就绪（瑞数挑战在浏览器里自动过）。
        NMPA 的 api_query / api_detail 都依赖这个首页会话，没它必败。

        首屏过不了瑞数（偶发渲染失败 / 被临时限流空白页）时，重启整个浏览器
        （全新进程 + 全新临时 profile，等价于一次全新启动）再试，最多 retries 次。
        两次重试之间按 30/60/90s **递增冷却**——瑞数多为短时 IP 限流，给窗口时间恢复，
        不能像以前那样在 1 分钟内连撞 3 次（必撞死）。
        """
        logger = get_logger()
        for attempt in range(retries):
            # 上一轮 restart() 可能失败并把 self.page 置空（新浏览器进程没拉起来），
            # 先确保有可用浏览器实例，否则 self.page.get 会抛 NoneType 直接崩循环。
            if self.page is None:
                try:
                    self.start()
                except Exception as e:  # noqa: BLE001
                    logger.warning("浏览器未就绪，重新启动失败（第 %d 次）：%s", attempt + 1, e)
            try:
                self.page.get(HOME_URL)
            except Exception as e:  # noqa: BLE001
                logger.warning("首页导航异常（第 %d 次）：%s", attempt + 1, e)
                self.page = None  # 标记失效，下一轮循环会重启或重新启动
            else:
                for _ in range(POLL_TIMEOUT):
                    time.sleep(POLL_INTERVAL)
                    try:
                        ok = self.page.run_js(
                            "return (function(){var r=document.querySelector('#home');"
                            "return (r&&r.__vue__)?'OK':'NO';})();")
                        if ok == "OK":
                            return
                    except Exception:
                        pass
            # 没过挑战：等限流窗口恢复 + 重启浏览器（全新进程）再试
            if attempt < retries - 1:
                wait = 30 * (attempt + 1)
                logger.warning("首页瑞数挑战未通过，%d 秒后重启浏览器重试（%d/%d）",
                               wait, attempt + 1, retries - 1)
                time.sleep(wait)
                try:
                    self.restart()
                except Exception as e:  # noqa: BLE001
                    logger.debug("重启浏览器失败：%s", e)
        raise RuntimeError("NMPA 首页加载失败：瑞数挑战未通过，检查网络或 EDGE_PATH")

    # ---------- 限速 ----------
    def _rate_limit(self):
        wait = self.interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    # ---------- NMPA pajax 接口（瑞数站点通用） ----------
    def _pajax(self, js):
        """执行一段调用 pajax 的 JS，轮询取回 JSON；失败返回 None。

        每个查询用独立令牌槽位（window['q...']）：pajax 异步响应可能乱序到达，
        共用同一变量会让后到的旧响应覆盖新响应导致数据串号（曾踩过坑）。
        """
        try:
            token = self.page.run_js(js)
            if not token:
                return None
            for _ in range(POLL_TIMEOUT):
                time.sleep(POLL_INTERVAL)
                res = self.page.run_js("return window['%s']||null;" % token)
                if res and res != "PENDING":
                    break
            else:
                return None
        except Exception:
            return None
        if not res or str(res).startswith("ERR"):
            return None
        try:
            return json.loads(res)
        except Exception:
            return None

    def api_query(self, keyword, item_id, page_num=1, page_size=20):
        """列表查询 api.queryList，返回 {"total": N, "list": [...]}；失败返回 None。"""
        self._rate_limit()
        kw = json.dumps(keyword, ensure_ascii=False)
        js = ("var token = 'q' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2);"
              "window[token] = 'PENDING';"
              "pajax.hasTokenGet(api.queryList, {itemId: '" + item_id + "', isSenior: 'N',"
              " searchValue: " + kw + ", pageNum: " + str(page_num) + ", pageSize: " + str(page_size) + "})"
              ".then(r => { window[token] = JSON.stringify(r); })"
              ".catch(e => { window[token] = 'ERR:' + e; });"
              "return token;")
        r = self._pajax(js)
        if r is None:
            return None
        try:
            d = r["data"]["data"]
            return {"total": d.get("total") or 0, "list": d.get("list") or []}
        except Exception:
            return None

    def api_detail(self, detail_id, item_id, retries=DETAIL_RETRIES, backoff=DETAIL_BACKOFF):
        """详情查询 api.queryDetail，返回正文字段 f3（str）；失败返回 None。

        注意滑动窗口节流——重试需等满 backoff（120s > 窗口长）。
        """
        for attempt in range(retries + 1):
            if attempt:
                time.sleep(backoff)
            self._rate_limit()
            if attempt:  # 上一轮没拿到正文，多半会话过期，先刷新会话再试
                self._refresh_session()
            js = ("var token = 'q' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2);"
                  "window[token] = 'PENDING';"
                  "pajax.hasTokenGet(api.queryDetail, {itemId: '" + item_id + "', id: '" + str(detail_id) + "'})"
                  ".then(r => { window[token] = JSON.stringify(r); })"
                  ".catch(e => { window[token] = 'ERR:' + e; });"
                  "return token;")
            r = self._pajax(js)
            if r is None:
                continue
            try:
                text = (r["data"]["data"]["detail"].get("f3") or "").strip()
                if text:
                    return text
            except Exception:
                continue
        return None

    def navigate(self, url, wait=6):
        """通用导航：chictr / wanfang 站点用。wait 秒给 SPA 渲染时间。"""
        self.page.get(url)
        time.sleep(wait)
        return self.page
