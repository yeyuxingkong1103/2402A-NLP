# 容器内启动脚本：把公共模块与工单01代码挂进 sys.path 并启动 Web 服务
# 工单编号：人工智能NLP-RAG 项目-金融问答系统部署
import sys  # 标准库：用于修改模块搜索路径 sys.path

# 把挂载进容器的公共模块目录加入搜索路径（放在最前，优先于 site-packages）
sys.path.insert(0, "/app/common")
# 把工单01的应用代码目录加入搜索路径（其中的 app.py 内含 Flask 实例）
sys.path.insert(0, "/app/app")

# 导入 Flask 应用对象（app.py 内含 Flask 实例化与 /api/ask 路由）；
# 因 import 语句在 sys.path 修改之后，故加 noqa 抑制 E402（模块级 import 不在文件顶部）告警
from app import app  # noqa: E402  （app.py 内含 Flask 实例化与 /api/ask 路由）

if __name__ == "__main__":
    # 容器内必须监听 0.0.0.0 才能被端口映射访问（127.0.0.1 只接受容器内部请求，外部无法访问）
    # port=8601：与 Dockerfile EXPOSE 及 compose 端口映射一致；debug=False：生产环境关闭调试模式
    app.run(host="0.0.0.0", port=8601, debug=False)
