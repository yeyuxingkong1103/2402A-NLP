# K8s 部署 + 多节点 + 高可用

> 工单编号: 人工智能 NLP-RAG-金融问答系统部署 (进阶)

## 一、Kubernetes 部署

### 1.1 Deployment

```yaml
# rag-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financial-rag
  replicas: 3
spec:
  selector:
    matchLabels:
      app: financial-rag
  template:
    metadata:
      labels:
        app: financial-rag
    spec:
      containers:
      - name: rag
        image: financial-rag:v10.0
        ports:
        - containerPort: 5008
        env:
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: llm-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: rag-data
          mountPath: /app/data
        - name: rag-cache
          mountPath: /app/cache
        livenessProbe:
          httpGet:
            path: /api/health
            port: 5008
          initialDelaySeconds: 15
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/health
            port: 5008
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: rag-data
        persistentVolumeClaim:
          claimName: rag-data-pvc
      - name: rag-cache
        emptyDir: {}
```

### 1.2 Service

```yaml
# rag-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: financial-rag
spec:
  type: LoadBalancer
  selector:
    app: financial-rag
  ports:
  - port: 5008
    targetPort: 5008
```

### 1.3 Ingress

```yaml
# rag-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: financial-rag
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  rules:
  - host: rag.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: financial-rag
            port:
              number: 5008
```

## 二、多节点负载均衡

```
                  ┌─ Pod (Node1) ─┐
  LoadBalancer ──┤─ Pod (Node2) ──┤── PV (持久化)
                  └─ Pod (Node3) ─┘
```

- 3 replicas: 高可用, 任意 1 个 Pod 挂掉不影响
- LoadBalancer: 自动负载均衡
- PV: 共享存储, 数据持久化

## 三、灰度发布

```bash
# 金丝雀发布: 先 10% 流量到新版本
kubectl apply -f rag-canary.yaml
# v10.0: 9 replicas
# v10.1: 1 replica (canary)

# 确认无问题后全量
kubectl apply -f rag-prod.yaml
# v10.1: 10 replicas
```

## 四、监控 (Prometheus + Grafana)

```yaml
# ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: financial-rag
spec:
  selector:
    matchLabels:
      app: financial-rag
  endpoints:
  - port: 5008
    path: /metrics
```

## 五、CI/CD

```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - run: docker build -t registry/rag:v10.0 .
    - run: docker push registry/rag:v10.0
    - run: kubectl apply -f k8s/
```

## 六、Docker vs K8s 选择

| 需求 | Docker Compose | Kubernetes |
|------|---------------|------------|
| 单机部署 | ✅ 简单 | 太重 |
| 开发环境 | ✅ | ✅ |
| 生产 1-2 台 | ✅ | 可选 |
| 生产 3+ 台 | ❌ | ✅ |
| 自动扩缩容 | ❌ | ✅ |
| 滚动更新 | ❌ | ✅ |
| 灰度发布 | ❌ | ✅ |
