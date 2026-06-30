"""Observability: OpenTelemetry GenAI tracing with pluggable backends.

Follows the 2025–2026 convergence on OpenTelemetry `gen_ai.*` semantic
conventions. Everything is optional and lazy — if the extras aren't installed,
calls are graceful no-ops. Telemetry honours an explicit opt-out.

Usage in an app::

    from agentx.observability import setup_tracing, get_callbacks
    setup_tracing("my-service")                     # instrument once at startup
    llm.invoke(msg, config={"callbacks": get_callbacks()})

Env:
  AGENTX_TELEMETRY=false          # global kill-switch (also OTEL_SDK_DISABLED=true)
  OTEL_EXPORTER_OTLP_ENDPOINT=... # send OTel spans here (Phoenix/Tempo/etc.)
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY [/ LANGFUSE_HOST]
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTRUMENTED = False


def telemetry_enabled() -> bool:
    """True unless explicitly disabled via AGENTX_TELEMETRY or OTEL_SDK_DISABLED."""
    if os.getenv("OTEL_SDK_DISABLED", "").lower() == "true":
        return False
    return os.getenv("AGENTX_TELEMETRY", "true").lower() not in ("false", "0", "no")


def setup_tracing(service_name: str = "agentx") -> bool:
    """Instrument LangChain with OpenTelemetry (OpenInference). Returns success.

    No-op (returns False) if telemetry is disabled or the optional extras
    (`agentx-kit[observability]`) aren't installed.
    """
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return True
    if not telemetry_enabled():
        logger.info("Telemetry disabled; skipping tracing setup.")
        return False
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.info("OpenTelemetry extras not installed; run `pip install 'agentx-kit[observability]'`.")
        return False

    try:
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        else:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        LangChainInstrumentor().instrument()
        _INSTRUMENTED = True
        logger.info("OpenTelemetry tracing enabled for '%s' (endpoint=%s).", service_name, endpoint or "console")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tracing setup failed: %s", exc)
        return False


def langfuse_callbacks() -> list:
    """Return [Langfuse CallbackHandler] if Langfuse is installed + configured."""
    if not telemetry_enabled():
        return []
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler  # older layout

            return [CallbackHandler()]
        except ImportError:
            logger.info("Langfuse not installed; `pip install langfuse` to enable.")
            return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse callback init failed: %s", exc)
        return []


def get_callbacks() -> list:
    """LangChain callbacks to pass via ``config={'callbacks': get_callbacks()}``."""
    return langfuse_callbacks()
