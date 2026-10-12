# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
分布式追踪：OpenTelemetry（可对接 Jaeger / Zipkin）。
未安装 opentelemetry 时降级为本地 Span 计时。
"""
import time


class Span:
    """本地降级 Span（无 OpenTelemetry 依赖）。"""

    def __init__(self, name):
        self.name = name
        self.start = time.perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.duration = time.perf_counter() - self.start


class Tracer:
    def __init__(self, service_name="rag"):
        self.service_name = service_name
        self.otel = None
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            provider = TracerProvider()
            provider.add_span_processor(BatchSpanProcessor(
                JaegerExporter(agent_host_name="localhost", agent_port=6831)))
            trace.set_tracer_provider(provider)
            self.otel = trace.get_tracer(service_name)
        except Exception as e:
            print("[tracing] OpenTelemetry 未安装，使用本地计时:", e)

    def start(self, name):
        if self.otel:
            return self.otel.start_as_current_span(name)
        return Span(name)


TRACER = Tracer()
