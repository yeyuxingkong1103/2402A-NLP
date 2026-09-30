import type { NextConfig } from "next";

/**
 * Next.js 配置。
 *
 * output: "standalone" —— 需求文档 3.4 明确要求生产环境使用 standalone 构建，
 * 产物自带精简的 node_modules，可直接放进 Docker 镜像，无需在容器内 npm install。
 *
 * 关于后端地址（重要）：
 * 原先用 rewrites() 把 /api/:path* 转发到 BACKEND_ORIGIN，但 rewrites 的目标是
 * **构建期**固化进 routes-manifest.json 的，运行时改环境变量无效——换后端地址必须重新构建。
 * 因此改为 app/api/[...path]/route.ts 运行时反向代理，BACKEND_ORIGIN 在**运行时**读取，
 * 同一份构建产物可在任意环境指向不同后端。
 */
const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
};

export default nextConfig;
