from typing import NamedTuple, Any, Literal, assert_never

from pydantic import BaseModel, TypeAdapter

AgentRole = Literal["writer", "doer", "solver"]

agent_roles = tuple(AgentRole.__args__)


Perspective = str | dict[str, Any]


class Example[I, O](NamedTuple):
    input: I
    output: O
    foreword: str | None = None
    commentary: str | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        def _dump(x):
            if isinstance(x, BaseModel):
                return x.model_dump(mode="json")
            return TypeAdapter(type(x)).dump_python(mode="json")
        out = {}
        if self.foreword:
            out["foreword"] = self.foreword
        out.update({
            "input": _dump(self.input),
            "output": _dump(self.output)
        })
        if self.commentary:
            out["commentary"] = self.commentary
        return out


class PromptingTask(NamedTuple):
    #this will get sorted alphabetically, so ctx, high, low, specific; keep that in mind if you modify it
    # we're lucky that the alphabetical order makes sense
    high_level: str
    low_level: str | None = None
    detailed: str | None = None
    context: str | None = None

    def to_prompt_dict(self, perspective: Perspective | None, examples: list[Example] | None) -> dict[str, Any]:
        out = {}
        if self.context:
            out["context"] = self.context
        out["high_level"] = self.high_level
        if perspective:
            out["perspective"] = perspective
        if self.low_level:
            out["low_level"] = self.low_level
        if self.detailed:
            out["detailed"] = self.detailed
        if examples:
            out["examples"] = [
                e.to_prompt_dict()
                for e in examples
            ]
        return out

def _behaviour(b: AgentRole):
    base = {
        "contract": {
            "precedence": "Contract rules override all other sections.",
        },
        "deliberation": {
            "permission": "Internal deliberation MAY be performed.",
            "visibility": "Internal deliberation MUST NOT appear in the response.",
        },
    }

    match b:
        case "writer":
            return {
                **base,
                "contract": {
                    **base["contract"],
                    "invariant": (
                        "Exactly one textual response is allowed. "
                        "Tool calling is forbidden."
                    ),
                },
                "output": (
                    "Natural language output is required. "
                    "Responses consisting only of analysis, deliberation, or meta-commentary are invalid. "
                    "If an output format is provided, the response MUST conform to it exactly."
                ),
            }

        case "doer":
            return {
                **base,
                "contract": {
                    **base["contract"],
                    "invariant": (
                        "Exactly one response is allowed. "
                        "The response MUST contain exactly one tool call "
                        "to the designated output tool."
                    ),
                    "fallback": (
                        "If no tool clearly applies, call the designated output tool anyway."
                    ),
                },
                "deliberation": {
                    **base["deliberation"],
                    "constraint": (
                        "After deliberation, you MUST immediately emit the tool call."
                    ),
                },
                "output": (
                    "Natural language output outside tools is forbidden. "
                    "The designated output tool is the only valid means of emitting results."
                ),
                "decisiveness": (
                    "If uncertain, choose an action and proceed. "
                    "An imperfect action is preferred over hesitation."
                ),
            }

        case "solver":
            return {
                **base,
                "contract": {
                    **base["contract"],
                    "invariant": (
                        "Exactly one response is allowed. "
                        "The response MUST contain exactly one tool call."
                    ),
                    "fallback": (
                        "If no tool clearly applies, call the designated output tool anyway."
                    ),
                },
                "deliberation": {
                    **base["deliberation"],
                    "constraint": (
                        "After deliberation, you MUST immediately emit the tool call."
                    ),
                },
                "output": (
                    "Natural language output outside tools is forbidden. "
                    "The designated output tool is the only valid means of emitting results."
                ),
                "decisiveness": (
                    "If uncertain, choose a tool and proceed. "
                    "An imperfect action is preferred over hesitation."
                ),
            }

        case _ as never:
            assert_never(never)


def system_prompt_dict(
    behaviour: AgentRole,
    task: PromptingTask,
    persona: str | None = None,
    perspective: Perspective | None = None,
    examples: list[Example] | None = None,
    input_format: str | None = None,
    output_format: str | None = None,
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
                "yaml": "hyphen-style",
                "empty_lists": {
                    "json": "[]",
                    "yaml": "[]",
                    "markdown": "- ... (single ellipsis-only hyphen-style item)"
                }
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
    out.update({"behaviour": _behaviour(behaviour)})
    if persona:
        out.update({
            "persona": persona
        })
    out.update({
        "task": task.to_prompt_dict(perspective, examples)
    })
    return out

def writer_system_prompt(*, persona: str, task: PromptingTask, 
                         perspective: Perspective | None = None, examples: list[Example] | None = None,
                         output_format: str | None = None):
    return system_prompt_dict(behaviour="writer", task=task, persona=persona,
                              perspective=perspective, examples=examples,
                              input_format="XML", output_format=output_format)

def doer_system_prompt(*, persona: str, task: PromptingTask,
                       perspective: Perspective | None = None, examples: list[Example] | None = None,):
    return system_prompt_dict(behaviour="doer", task=task, persona=persona,
                              perspective=perspective, examples=examples,
                              input_format="XML", output_format="structured tool call")

def solver_system_prompt(*, persona: str, task: PromptingTask,
                         perspective: Perspective | None = None, examples: list[Example] | None = None,):
    return system_prompt_dict(behaviour="solver", task=task, persona=persona,
                              perspective=perspective, examples=examples,
                              input_format="XML", output_format="structured tool call")
