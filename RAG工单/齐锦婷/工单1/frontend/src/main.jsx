import React from "react";
import { createRoot } from "react-dom/client";
import { App as AntApp, ConfigProvider } from "antd";
import zhCN from "antd/locale/zh_CN";
import App from "./App";

createRoot(document.getElementById("root")).render(<ConfigProvider locale={zhCN}><AntApp><App /></AntApp></ConfigProvider>);
