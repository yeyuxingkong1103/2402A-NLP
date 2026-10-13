# Prometheus integration

Expose `observability.metrics.METRICS.exposition()` from the deployment's `/metrics` handler and scrape it every 15 seconds. Wrap expensive retrieval/generation work in `observability.concurrency.API_REQUEST_LIMITER.slot()` to bound in-flight work and prevent unbounded memory growth.
