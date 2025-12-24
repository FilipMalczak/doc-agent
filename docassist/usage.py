from collections import defaultdict
from typing import Literal

from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span

UsageClass = Literal["all", "sampled", "unsampled"]

class UsageCollector(SpanProcessor):
    def __init__(self):
        self.by_model = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "calls": 0,
                }
            )
        )

    def on_end(self, span: Span) -> None:
        attrs = span.attributes or {}
        # FIXME THESE ARE WRONG

        model = attrs.get("gen_ai.response.model")
        if not model:
            return  # ignore non-LLM spans

        def bump(x: UsageClass):
            self.by_model[x][model]["prompt_tokens"] += attrs.get(
                "gen_ai.usage.input_tokens", 0
            )
            self.by_model[x][model]["completion_tokens"] += attrs.get(
                "gen_ai.usage.output_tokens", 0
            )
            self.by_model[x][model]["reasoning_tokens"] += attrs.get(
                "gen_ai.usage.details.reasoning_tokens", 0
            )
            self.by_model[x][model]["calls"] += 1

        bump("all")
        bump("sampled" if attrs.get("sampling.action") == "reuse" else "unsampled")

    # Required no-op hooks
    def on_start(self, span, parent_context): ...
    def shutdown(self): ...
    def force_flush(self, timeout_millis: int = 30000): ...

USAGE_COLLECTOR = UsageCollector()