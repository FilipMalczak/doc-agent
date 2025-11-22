from typing import Annotated, NamedTuple, Any, Awaitable

from pydantic_ai import Agent, TextOutput, ModelRetry

from docassist.config import CONFIG
from docassist.simple_xml import to_simple_xml
from docassist.system_prompts import simple_xml_system_prompt, PromptingTask

MD_CODE = "```"

def sanitize_markdown(txt: str) -> str:
    """
    Sometimes the model responds with

        ```yaml
        whatever
        ```

    If the format isn't yaml but md or markdown, this strips the code formatting. Otherwise, it forces a model retry.
    """
    if txt.startswith(MD_CODE):
        txt = txt[len(MD_CODE):]
        format, newline, rest = txt.partition("\n")
        format = format.strip()
        if format and format.lower() not in {"md", "markdown"}:
            raise ModelRetry(f"You should respond with Markdown! You replied with Markdown code block of {format} instead!")
        if txt.endswith(MD_CODE):
            txt = txt[:-len(MD_CODE)]
    return txt

Markdown = Annotated[str, "Text in Markdown format"]
MarkdownOutput = TextOutput(sanitize_markdown)

class SourceFile(NamedTuple):
    path: str
    source_language: str
    source_type: str
    content: str

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.source_language,
            "type": self.source_type,
            "content": self.content
        }

note_taker = Agent(
    name="single-file-note-taker",
    model=CONFIG.model,
    output_type=MarkdownOutput,
    output_retries=3,
    system_prompt=simple_xml_system_prompt(
        PromptingTask(
            high_level="take notes from the input file",
            low_level="take notes from the perspective of the user of this project",
            detailed="take notes that can be later used to prepare user-facing documentation of the project "
                     "that the input file is part of",
            context="you're reading the whole project for the first time; you need to extract the useful information for later usage"

        ),
        "note taker",
        output_format="Markdown"
    )
)



async def take_notes(source_file: SourceFile) -> Markdown:
    #fixme start tracking usage
    #todo sanitize_markdown that will remove ```yaml(...) ```; https://ai.pydantic.dev/api/output/#pydantic_ai.output.TextOutput
    return (await note_taker.run(to_simple_xml(source_file.to_prompt_dict()))).output

# https://chatgpt.com/share/691ca6cf-8d44-8005-a659-7c456ac8581b
# use miniperscache over nodes
# when returning an index from a node, return Snapshot instead
# Snapshot should have snapshot_id and instance fields and get() method
# modify its __(g|s)setstate__ to omit instance in serialization and flush it to file on write (getstate)/load on get