# 工单编号：人工智能NLP-RAG-金融问答系统部署
#
# RTMP 流媒体服务，用于验收标准「容器网络配置正确，能够与其他服务
# （如 RTMP 服务器）正常通信」里的对端。
#
# 为什么自己构建而不是拉现成镜像：
#   alfg/nginx-rtmp、tiangolo/nginx-rtmp、jasonrivers/nginx-rtmp 都试过，
#   本机 registry 加速器一律返回 403 Forbidden（见《实现步骤与问题记录》问题 8）。
#   Alpine 官方源里有现成的 nginx-mod-rtmp 包，自己装一下即可，
#   镜像只有十几 MB，而且**部署到目标服务器时不依赖任何第三方镜像**，
#   比拉一个来路不明的第三方镜像更可控。
#
# 构建：
#   docker build -f 部署/rtmp/Dockerfile -t rag-rtmp:1.0 部署/rtmp

FROM alpine:3.20

# 换成国内源，apk 拉包快很多（官方 dl-cdn 在国内经常几十 KB/s）
RUN sed -i 's|dl-cdn.alpinelinux.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apk/repositories \
    && apk add --no-cache nginx nginx-mod-rtmp \
    && mkdir -p /var/log/nginx /var/lib/nginx/tmp \
    && ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log

COPY nginx.conf /etc/nginx/nginx.conf

# 1935 是 RTMP 协议默认端口，80 提供状态页与健康检查
EXPOSE 1935 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1/health >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
