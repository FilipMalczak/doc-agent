from typing import NamedTuple, Any

from docassist.simple_xml import to_simple_xml


class PromptingTask(NamedTuple):
    #this will get sorted alphabetically, so ctx, high, low, specific; keep that in mind if you modify it
    # we're lucky that the alphabetical order makes sense
    high_level: str
    low_level: str | None = None
    detailed: str | None = None
    context: str | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        out = {}
        if self.context:
            out["context"] = self.context
        out["high_level"] = self.high_level
        if self.low_level:
            out["low_level"] = self.low_level
        if self.detailed:
            out["detailed"] = self.detailed
        return out

def system_prompt_dict(
        task: PromptingTask,
        persona: str | None = None,
        input_format: str | None = None,
        output_format: str | None = None,
        turbo: bool = False, #fixme
) -> dict[str, Any]:
    out = dict()
    formats = {}
    if input_format:
        formats["input"] = input_format
    if output_format:
        formats["output"] = output_format
    if formats:
        out.update({
            "format": formats
        })
    out.update({
        "formatting_rules": {
            "quotes": "always use double quotes for strings, no matter the format",
            "indentation": "optional, recommended",
            "empty_values": "omit, unless semantically important or required by schema",
            "lists": {
                "markdown_bullet_points": "hyphen-style",
                "yaml": "hyphen-style"
            },
            "multiline_text": {
                "yaml": "Prefer `: |` style"
            }
        },
        "constraints": {
            "hallucinations": "forbidden without exception",
            "invent_data": "forbidden, unless explicitly asked to",
            "modify_content": "only structure and wording, never meaning",
            "modify_formatting": "allowed, as long as other constraints and rules satisfied",
            "match_output_schema": "required if schema available",
            "match_output_format": "required if `format.output` available",
        }
    })
    if turbo:
        out.update({
            "behaviour": {
                "contract": {
                    "invariant": "Each response MUST contain exactly one tool call. Responses without a tool call are "
                                 "invalid.",
                    "fallback": "If no tool clearly applies, call final_result anyway." #todo tool name may differ
                },
                "reasoning": {
                    "permission": "Internal reasoning MAY be performed.",
                    "constraint": "Reasoning is strictly limited. After reasoning, you MUST immediately call a tool.",
                    "prohibition": "Reasoning alone is forbidden as a response."
                },
                "output": "Natural language output outside tools is forbidden. Responses consisting only of thinking or "
                          "reasoning are invalid. `final_result` tool is the only valid means of emitting output.", #todo again, tool name
                "decisiveness": "If uncertain, choose a tool and proceed. An imperfect action is preferred over hesitation."
            }
        })
    if persona:
        out.update({
            "persona": persona
        })
    out.update({
        "task": task.to_prompt_dict()
    })
    return out

def simple_xml_system_prompt(task: PromptingTask,
        persona: str | None = None,
        input_format: str | None = None,
        output_format: str | None = None,
        turbo: bool = False, #fixme
) -> str:
    return to_simple_xml(system_prompt_dict(task, persona, input_format, output_format, turbo))
