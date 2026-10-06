# 法衡 Desktop Client

这是一个很薄的 Electron 桌面壳，只负责打开法衡后端的 Web UI。
当前打包版本不会硬编码收件人的本机地址；首次打开时请在连接失败页或设置里填写后端地址。

## 运行

```powershell
cd desktop-client
npm install
npm start -- --server=http://127.0.0.1:7294
```

## 打包成 exe

```powershell
cd desktop-client
npm install
npm run dist:win
```

生成的 portable exe 会放在 `desktop-client/dist`，文件名是 `法衡.exe`。

## 发给别人

- 直接发送 `dist/法衡.exe`，不需要发送 `node_modules` 或源码。
- 对方电脑需要能访问你配置的后端地址，并且服务端电脑和 Docker 服务保持运行。
- 如果服务端 IP 变化，对方可以用 `--server=http://实际IP:7294` 启动，或在连接失败页面修改地址。
- 这是未签名的便携版 EXE，Windows SmartScreen 可能首次提示风险，需要选择允许运行。

## 备注

- 这个客户端不包含模型、MySQL、Redis 或 Milvus。
- 先把后端 Docker 服务启动好，再让桌面壳连到后端地址。
- 如果后端不在本机，把 `--server=` 改成实际地址即可。
