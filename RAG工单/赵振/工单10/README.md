# 工单10：金融问答系统Docker部署
作者：赵振。配套工单9必须放在相邻目录；不增加Python代码。
已在本机Docker Desktop Linux引擎实际构建、运行并验证，并非仅提供Docker配置。
当前服务：http://127.0.0.1:8509 。本轮没有远程服务器地址，验证范围为本机。

## 最简单启动
Windows先启动Docker Desktop，双击本目录“启动Docker.bat”。脚本创建命名数据卷和网络，使用Compose构建并启动。
第一次构建需要联网安装依赖；浏览器打开时若服务仍在建索引，等健康检查通过后刷新。
Ollama可选：本机启动Ollama并下载deepseek-r1:7b。默认摘录模式不依赖Ollama。

## 手动docker run（在全新环境）
在“赵振”目录运行：
```powershell
docker network create zhaozhen-finance-net
docker volume create zhaozhen-finance-data
docker build -f .\工单10\Dockerfile -t zhaozhen-finance:workorder10 .
docker run -d --name zhaozhen-finance-10 --restart unless-stopped --network zhaozhen-finance-net -p 8509:8000 -v zhaozhen-finance-data:/app/data --add-host host.docker.internal:host-gateway -e OLLAMA_URL=http://host.docker.internal:11434 zhaozhen-finance:workorder10
docker logs zhaozhen-finance-10
docker inspect zhaozhen-finance-10 --format '{{.State.Health.Status}}'
```
若已有同名容器或8509端口已被占用，不再重复运行。当前交付机器已由Compose管理，同名手动测试容器已经停止并保留为run-verified。

## Compose管理
在工单10目录运行：
```powershell
docker compose up -d --build
docker compose ps
docker compose logs
docker compose stop
docker compose start
```
compose.yaml显式设置英文项目名，外部命名卷和网络由启动脚本创建。

## 数据和网络
持久数据位于命名卷zhaozhen-finance-data，对应/app/data，包含pages.json、graph.json及后续上传的PDF。
镜像预置2812页；新建空卷首次启动复制初始数据。已有卷不会因为重建镜像自动覆盖数据。
独立上传测试卷的2815页在重启后仍保留。第二个容器通过只读共享卷读取2812页，并通过容器名访问主服务健康接口成功。
主服务可访问host.docker.internal上的Ollama；Linux服务器部署时需配置实际可达的模型服务地址。
未部署RTMP服务；本工单的跨服务通信采用容器HTTP接口和Ollama验证。

## 验证材料与问题记录
见测试记录.md、部署验证.json、功能验证.json、容器最终状态.json及结果图。
真实测试包含构建、docker run、Compose、健康检查、API问答、损坏PDF、无效输入、真实PDF上传、重启持久化、共享卷和容器网络、8并发40请求。
无效输入返回400，模型不可用返回503。只实测短时运行，没有证明长期高可用。
曾遇到中文目录默认Compose项目名问题，已显式设置name；Docker命名管道权限通过当前环境允许的授权执行解决。
旧测试容器和卷保留，没有执行删除；独立上传测试容器已停止。
浏览器截图未取得；结果图、讲解视频是实际日志与JSON结果的展示，不是网页截图或现场录屏。

官方参考：[docker run](https://docs.docker.com/reference/cli/docker/container/run/)、[端口发布](https://docs.docker.com/get-started/docker-concepts/running-containers/publishing-ports/)、[Desktop网络](https://docs.docker.com/desktop/features/networking/networking-how-tos/)。
