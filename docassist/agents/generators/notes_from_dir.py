from dataclasses import field
from os.path import basename, join, dirname
from pathlib import Path
from typing import NamedTuple, Self, Iterable

from pydantic import BaseModel
from pydantic_ai import Agent

from docassist.agents.generators.notes_from_file import MarkdownOutput
from docassist.config import CONFIG
from docassist.index.document import DirNoteMeta
from docassist.index.protocols import Document
from docassist.subjects import CodeFilePath, EntryType
from docassist.system_prompts import simple_xml_system_prompt, PromptingTask

RelativePath = str
Filename = str
Dirname = str

class DirectoryDescriptor(NamedTuple):
    relative_path: RelativePath
    files: dict[str, CodeFilePath] = field(default_factory=dict)
    directories: dict[str, Self] = field(default_factory=dict)

    def depth_first(self) -> Iterable[tuple[RelativePath, list[Filename], list[Dirname]]]:
        for s in self.directories.values():
            yield from s.depth_first()
        yield (
            self.relative_path,
            list(self.files.keys()),
            list(self.directories.keys())
        )


def structurize(files: Iterable[CodeFilePath]) -> DirectoryDescriptor:
    root = DirectoryDescriptor("", {}, {}) #fixme defaults ffs
    def _get(dir_path: str) -> DirectoryDescriptor:
        current = root

        parts = Path(dir_path).parts
        path_up_to_now = None
        for p in parts:
            if path_up_to_now is None:
                path_up_to_now = p
            else:
                path_up_to_now = join(path_up_to_now, p)
            if p not in current.directories:
                current.directories[p] = DirectoryDescriptor(path_up_to_now, {}, {}) #fixme wth wont defaults work?
            current = current.directories[p]
        return current

    for f in files:
        base = dirname(f.path)
        d = _get(base)
        fn = basename(f.path)
        d.files[fn] = f
    return root

class SubjectNotes(BaseModel):
    subject_path: str
    subject_type: EntryType
    notes_format: str = "markdown"
    content: str

class LowerLevelNotes(BaseModel):
    lower_level_notes: list[SubjectNotes]

dir_note_taker = Agent(
    name="note taker that handles a whole directory",
    model=CONFIG.model,
    output_type=MarkdownOutput,
    output_retries=3,
    system_prompt=simple_xml_system_prompt(
        persona="note taker",
        task=PromptingTask(
            high_level="take notes from the files in a directory",
            low_level="use previously prepared file-level notes and notes on subdirectories generated the same way "
                      "as what you're doing now and prepare directory-level extract",
            detailed="take notes that can be later used to prepare user-facing documentation of the project "
                     "that the input file is part of",
            context="you've read the whole project once, you're compiling your file-level notes into directory-level notes"
        ),
        output_format="Markdown"
    )
)


def doc_to_notes(d: Document) -> SubjectNotes:
    return SubjectNotes(
        subject_path=d.metadata.subject_path,
        subject_type=d.metadata.subject_type,
        content=d.content
    )

def dir_notes_input(dir_children_notes: list[Document[DirNoteMeta]]) -> tuple[dict[str, SubjectNotes], list[CodeFilePath]]:
    def doc_to_path(d: Document) -> CodeFilePath:
        return CodeFilePath(d.metadata.subject_path, language=d.metadata.subject_language)
    l, r = {}, []
    for d in dir_children_notes:
        l[d.metadata.subject_path] = doc_to_notes(d)
        r.append(doc_to_path(d))
    return l, r

