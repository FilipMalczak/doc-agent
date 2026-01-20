from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span

from docassist.config import CONFIG
from docassist.models import TokenInference

class UsageCollector(SpanProcessor):

    def on_end(self, span: Span) -> None:
        attrs = span.attributes or {}
        # FIXME THESE ARE WRONG; or are they?

        model = attrs.get("gen_ai.response.model")
        if not model:
            return  # ignore non-LLM spans
        CONFIG.model_broker.get_model_profile(model).observe(
            TokenInference(
                input=attrs.get("gen_ai.usage.input_tokens", 0),
                output=attrs.get("gen_ai.usage.output_tokens", 0),
                #todo add reasoning, etc
                sampled=attrs.get("sampling.action", "") == "reuse"
            )
        )

    # Required no-op hooks
    def on_start(self, span, parent_context): ...
    def shutdown(self): ...
    def force_flush(self, timeout_millis: int = 30000): ...

USAGE_COLLECTOR = UsageCollector()