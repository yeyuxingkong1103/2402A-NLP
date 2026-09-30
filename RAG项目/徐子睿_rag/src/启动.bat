@echo off
rem ============================================================
rem 启动.bat —— Windows 一键启动脚本（backend 那条主线）
rem
rem 用法：双击本文件，或在命令行执行 启动.bat
rem 作用：启动 backend/server.py 的 FastAPI 服务，然后自动打开浏览器
rem
rem 注意它启动的是 backend.server:app（老主线：单角色知识库 + 原生前端），
rem 不是 src.api.main:app（新架构：多用户多角色 + JWT）。
rem 要启动新架构请用 scripts/start.sh 或直接：
rem     python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8902
rem     （新架构用 8902，与本脚本的 8901 分开，两者才能同时运行）
rem
rem 前置条件：已安装依赖（pip install -r requirements.txt）；
rem 完整检索功能还需要 Milvus 与 Ollama 已启动（见 README 的排障 FAQ）。
rem
rem 本脚本做了三道防护，都是踩过坑之后加的：
rem   1. 启动前检查端口是否被占，被占就提示并询问是否结束旧进程
rem   2. 轮询端口直到服务真正就绪，才打开浏览器
rem   3. 服务跑在 cmd /k 窗口里，报错不会一闪而过
rem
rem 【编码说明】本文件保存为 GBK(ANSI) 编码，不要改成 UTF-8。
rem   原因：cmd.exe 在中文 Windows 上按 GBK 读取 .bat。UTF-8 的汉字被按 GBK
rem   解读后，字节对里可能凑出重定向、管道、与号这些操作符，把一行注释劈成
rem   命令去执行（实测会报"xxx 不是内部或外部命令"）。
rem   同理，任何 .bat 里的 rem 注释也不要写这些操作符字符本身。
rem ============================================================
setlocal
rem cd /d "%~dp0" 切到本文件所在目录（%~dp0 = 批处理文件所在路径）。
rem 必须先切换，否则从其他目录双击运行时，后面的相对路径全部找不到。
cd /d "%~dp0"

rem 端口集中定义，下面统一引用，避免改一处漏一处
set "PORT=8901"
rem 等待服务就绪的最长秒数
set "WAIT_MAX=30"

echo ============================================
echo   RAG knowledge base starting...
echo   http://127.0.0.1:%PORT%
echo ============================================

rem ------------------------------------------------------------
rem 防护 1：启动前检查端口是否已被占用
rem   为什么必须查：残留的 uvicorn（比如上次没关干净）会一直占着端口，
rem   新进程 bind 失败。而 uvicorn 会先打印 Application startup complete
rem   再报错，且报错在新窗口里一闪而过 —— 看起来像启动成功了，
rem   实际浏览器打开的是旧应用（症状是 /health 之类返回 404）。
rem   netstat 的第 5 列是 PID；只取 LISTENING，不把 TIME_WAIT 行（PID 为 0）算进来。
rem ------------------------------------------------------------
set "OCCUPIED="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do set "OCCUPIED=%%P"
if not defined OCCUPIED goto START

echo.
echo [警告] 端口 %PORT% 已被进程 %OCCUPIED% 占用。
echo         多半是上次没关干净的 uvicorn —— 这种情况下新服务会启动失败，
echo         而浏览器打开的还是旧应用，看起来像启动不了却看不到报错。
echo.
rem /T 20 /D N：20 秒无输入则默认选 N（不自动杀进程，杀进程这种事不替用户做主）
choice /C YN /T 20 /D N /M "是否先结束该进程再启动（20 秒无输入默认 N）"
if errorlevel 2 (
  echo.
  echo 已取消启动。你可以自己确认后处理，或先关掉占用 %PORT% 的程序再重试。
  pause
  exit /b 1
)
taskkill /PID %OCCUPIED% /F
rem 等约 2 秒让操作系统真正释放端口，紧接着启动才不会偶发 bind 失败。
rem 用 ping 而不是 timeout 做等待：PATH 上若有其他工具提供的 timeout，
rem 会遮蔽 Windows 自带的 timeout.exe（实测报 invalid time interval /t）。
rem ping 在 System32 下，且不会与其他工具重名。
ping -n 3 127.0.0.1 >nul

:START
rem ------------------------------------------------------------
rem 防护 3：服务跑在 cmd /k 窗口里
rem   /k 让命令结束后窗口不关闭，这样依赖缺失、端口冲突之类的报错
rem   会留在屏幕上可读 —— 直接 start python 的话，报错会随窗口一起消失。
rem ------------------------------------------------------------
start "RAG" cmd /k python -m uvicorn backend.server:app --host 127.0.0.1 --port %PORT%

rem ------------------------------------------------------------
rem 防护 2：轮询端口，等服务真的就绪再开浏览器
rem   原来是盲等 3 秒就开浏览器：服务如果启动失败，浏览器照样打开，
rem   用户看到的是错误页或旧应用，于是得出"启动不了"的结论。
rem   这里改成每秒探一次、最多等 WAIT_MAX 秒；等不到就明确报错并提示去看 RAG 窗口。
rem   用 goto 循环而不是 for 循环：goto 每次重新解析整行，WAITED 每次都能取到最新值，
rem   不必依赖延迟展开（delayed expansion）。
rem ------------------------------------------------------------
set /a WAITED=0
:WAIT
rem 每次循环先等约 1 秒（ping 的等待技巧同上）
ping -n 2 127.0.0.1 >nul
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 goto READY
set /a WAITED+=1
if %WAITED% lss %WAIT_MAX% goto WAIT

echo.
echo [错误] 等了 %WAIT_MAX% 秒，%PORT% 仍未就绪。
echo        请查看新打开的 RAG 窗口里的报错信息，常见原因：
echo          - 依赖没装齐（先执行 python -m pip install -r requirements.txt）
echo          - 端口被别的程序占用
echo          - 当前目录下缺少项目文件（本脚本要先切到自己所在目录）
pause
exit /b 1

:READY
start http://127.0.0.1:%PORT%
