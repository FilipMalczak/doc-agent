from typing import NamedTuple, Any, Literal, assert_never

from pydantic import BaseModel, TypeAdapter

AgentRole = Literal["writer", "doer", "solver"]

agent_roles = tuple(AgentRole.__args__)


Perspective = str | dict[str, Any]


class Example[I, O](BaseModel):
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


class PromptingTask(BaseModel):
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

def _behaviour(b: AgentRole, empty_allowed: bool, may_not_know: bool):
    tool_calling = False
    text_allowed = False
    EMPTY = {
        "no_answer": "if a problem, task or question you're facing has no meaningful response or answer, "
                     "`indicate it with a `empty_result` tool call; don't use this to indicate your lack of "
                     "capability but rather 'no value' response",
    }
    DONT_KNOW = {
        "cannot_answer": "if you're unable to produce the response or answer, even though one should be possible "
                         "to come up wth, indicate it with a `i_dont_know` tool call; use it to indicate that "
                         "you cannot achieve the result; do not use this to indicate lack of meaningful result",
        }
    out = {}
    if b == "writer":
        text_allowed = True
        if empty_allowed or may_not_know:
            tool_calling = True
            out.update({
                "tool_calling": {
                    "allowed": "to indicate special response cases",
                    "text_response": "if your response contains a tool call, any assistant text message will be ignored"
                },
            })
            if empty_allowed:
                out.update(EMPTY)
            if may_not_know:
                out.update(DONT_KNOW)
            out.update({
                "result": "if you're able to produce a meaningful result, emit it as a standard non-empty assistant text "
                          "message; indicate otherwise with the appropriate tools",
            })
        else:
            out.update({
                "tool_calling": {
                    "forbidden": "without exception"
                },
                "result": "you must emit a single standard non-empty assistant text message",
                "no_result": "forbidden; each and every response must include non-empty assistant message"
            })
    else:
        tool_calling = True
        if b == "doer":
            out.update({
                "tool_calling": {
                    "required": "to indicate response",
                    "text_response": "any assistant text message will be ignored"
                },
            })
            if empty_allowed or may_not_know:
                if empty_allowed:
                    out.update(EMPTY)
                if may_not_know:
                    out.update(DONT_KNOW)
                out.update({
                    "result": "if you're able to produce a meaningful result, emit it using `final_result` tool; indicate "
                              "otherwise with the appropriate tools; you must emit exactly one response; that response "
                              "must contain a tool call to one of the tools indicating response or lack of one",
                })
            else:
                out.update({
                    "result": "you must emit a single `final_result` tool call; no other response (outside of thinking) "
                              "is allowed",
                })
        elif b == "solver":
            out.update({
                "tool_calling": {
                    "allowed": "to obtain additional data or to indicate response",
                    "text_response": "any assistant text message will be ignored",
                    "forbidden": "using the same tool twice with the same arguments is forbidden"
                },
            })
            if empty_allowed:
                out.update(EMPTY)
            if may_not_know:
                out.update(DONT_KNOW)
            out.update({
                "decisiveness": "emit the response as soon as you can anchor it in the supporting data; you might take "
                                "an additional step or two to make sure, but don't overresearch; never emit a response"
                                "without supporting it with retrieved knowledge",
                "research": "required; any answer, response or result you give must be anchored in supprting data you "
                            "retrieve via tools; results with no supporting evidence are invalid; refer to the evidence"
                            "in the explanation",
                "inherent_knowledge": "you might use the knowledge you had without using tools to prepare hypothesis, "
                              "conduct reasoning, planning and so on; never use it to support your decision or result",
                "result": "if you're able to produce a meaningful result, emit it using `final_result` tool; indicate "
                          "otherwise with the appropriate tools",
            })
        else:
            assert False, f"WTF is {b} role???"

    must_include = []
    if text_allowed:
        must_include.append("non-empty text message")
    if tool_calling:
        if must_include:
            must_include.append("or a")
        must_include.append("tool call")
    must_include = [ "exactly one" if text_allowed else "at least one" ] + must_include
    out.update({
        "thinking": {
            "allowed": "internal reasoning can be performed before producing any kind of input",
            "forbidden": "you cannot emit a message that consists only of thinking section; "
                         "each response must include "+(" ".join(must_include))
        }
    })
    return out

def system_prompt_dict(
    behaviour: AgentRole,
    empty_allowed: bool, may_not_know: bool,
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
    out.update({"behaviour": _behaviour(behaviour, empty_allowed, may_not_know)})
    if persona:
        out.update({
            "persona": persona
        })
    out.update({
        "task": task.to_prompt_dict(perspective, examples)
    })
    return out

def writer_system_prompt(*, empty_allowed: bool, may_not_know: bool,
                         persona: str, task: PromptingTask,
                         perspective: Perspective | None = None, examples: list[Example] | None = None,
                         output_format: str | None = None):
    return system_prompt_dict(behaviour="writer", empty_allowed=empty_allowed, may_not_know=may_not_know,
                              task=task, persona=persona,
                              perspective=perspective, examples=examples,
                              input_format="XML", output_format=output_format)

def doer_system_prompt(*, empty_allowed: bool, may_not_know: bool,
                       persona: str, task: PromptingTask,
                       perspective: Perspective | None = None, examples: list[Example] | None = None,):
    return system_prompt_dict(behaviour="doer", empty_allowed=empty_allowed, may_not_know=may_not_know,
                              task=task, persona=persona,
                              perspective=perspective, examples=examples,
                              input_format="XML", output_format="structured tool call")

def solver_system_prompt(*, empty_allowed: bool, may_not_know: bool,
                         persona: str, task: PromptingTask,
                         perspective: Perspective | None = None, examples: list[Example] | None = None,):
    return system_prompt_dict(behaviour="solver", empty_allowed=empty_allowed, may_not_know=may_not_know,
                              task=task, persona=persona,
                              perspective=perspective, examples=examples,
                              input_format="XML", output_format="structured tool call")
