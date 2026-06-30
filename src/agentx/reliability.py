"""Reliability: provider fallbacks, retries, and token/cost budgets.

- ``build_resilient_chat`` wraps a primary model with retries and sequential
  fallbacks to other providers/models (à la pydantic-ai's FallbackModel +
  the 'circular fallback' pattern from production templates).
- ``UsageLimits`` + ``UsageTracker`` enforce per-run request/token/cost budgets
  to stop runaway agent loops (à la pydantic-ai ``UsageLimits``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .providers import get_chat_model

logger = logging.getLogger(__name__)


def build_resilient_chat(
    provider: str | None = None,
    model: str | None = None,
    fallbacks: list[tuple[str, str]] | None = None,
    retries: int = 2,
    **kwargs,
):
    """Return a chat model with retry + ordered provider/model fallbacks.

    ``fallbacks`` is a list of ``(provider, model)`` tried in order if the
    primary fails. Built on LangChain's ``.with_retry()`` / ``.with_fallbacks()``.
    """
    primary = get_chat_model(provider, model, **kwargs)
    if retries and retries > 1:
        primary = primary.with_retry(stop_after_attempt=retries, wait_exponential_jitter=True)
    if fallbacks:
        alts = []
        for fb_provider, fb_model in fallbacks:
            try:
                alt = get_chat_model(fb_provider, fb_model, **kwargs)
                alts.append(alt.with_retry(stop_after_attempt=retries, wait_exponential_jitter=True))
            except Exception as exc:  # noqa: BLE001 - a missing extra shouldn't kill the chain
                logger.warning("Skipping fallback %s/%s: %s", fb_provider, fb_model, exc)
        if alts:
            primary = primary.with_fallbacks(alts)
    return primary


class UsageLimitExceeded(RuntimeError):
    """Raised when a configured request/token/cost budget is exceeded."""


@dataclass
class UsageLimits:
    max_requests: int | None = None
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None
    # USD per 1K tokens (total). Override per your provider/model pricing.
    price_per_1k_tokens: float = 0.0


@dataclass
class UsageTracker:
    """Accumulates usage and enforces ``UsageLimits``.

    Call :meth:`record` with token counts after each model call (or attach
    :meth:`as_callback` to LangChain). Raises ``UsageLimitExceeded`` on breach.
    """

    limits: UsageLimits = field(default_factory=UsageLimits)
    requests: int = 0
    total_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return round(self.total_tokens / 1000.0 * self.limits.price_per_1k_tokens, 6)

    def record(self, tokens: int = 0) -> None:
        self.requests += 1
        self.total_tokens += max(0, int(tokens))
        self._enforce()

    def _enforce(self) -> None:
        lim = self.limits
        if lim.max_requests is not None and self.requests > lim.max_requests:
            raise UsageLimitExceeded(f"max_requests exceeded ({self.requests} > {lim.max_requests})")
        if lim.max_total_tokens is not None and self.total_tokens > lim.max_total_tokens:
            raise UsageLimitExceeded(f"max_total_tokens exceeded ({self.total_tokens} > {lim.max_total_tokens})")
        if lim.max_cost_usd is not None and self.cost_usd > lim.max_cost_usd:
            raise UsageLimitExceeded(f"max_cost_usd exceeded (${self.cost_usd} > ${lim.max_cost_usd})")

    def as_callback(self):
        """Return a LangChain callback handler that records token usage."""
        from langchain_core.callbacks import BaseCallbackHandler

        tracker = self

        class _UsageCallback(BaseCallbackHandler):
            def on_llm_end(self, response, **kwargs):  # noqa: ANN001
                tokens = 0
                try:
                    usage = (response.llm_output or {}).get("token_usage") or {}
                    tokens = usage.get("total_tokens", 0)
                    if not tokens:  # some providers attach usage to message metadata
                        for gen in getattr(response, "generations", []) or []:
                            for g in gen:
                                meta = getattr(getattr(g, "message", None), "usage_metadata", None)
                                if meta:
                                    tokens += meta.get("total_tokens", 0)
                except Exception:  # noqa: BLE001
                    tokens = 0
                tracker.record(tokens)

        return _UsageCallback()
