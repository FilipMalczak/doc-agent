from dataclasses import field, dataclass
from functools import cached_property
from typing import Self, Iterable, Literal, assert_never
from uuid import uuid4

from docassist.index.document import ChunkVariant, NoteMeta, NoteChunkMeta, NoteSimpleChunkMeta, NoteDerivedChunkMeta, \
    NoteChunkVariant, DerivedChunkVariant
from docassist.index.protocols import Document
from docassist.index.utils import embed_metadata


@dataclass
class MarkdownChapter:
    title: str | None
    level: int
    parent: Self | None = field(default=None, repr=False, compare=False, hash=False)
    content: list[str] = field(default_factory=list)
    children: list[Self] = field(default_factory=list)

    def render_lines(self) -> list[str]:
        result = []
        if self.level:
            result.append("#"*self.level + self.title)
        result.extend(self.content)
        for child in self.children:
            result.extend(child.render_lines())
        return result

    def render(self) -> str:
        return "\n".join(self.render_lines())

    @classmethod
    def root(cls, content: list[str] | None = None, children: list[Self] | None = None) -> Self:
        return cls(None, 0, None, content or [], children or [])

    @classmethod
    def _unprefix_header(cls, line: str) -> tuple[int, str]:
        stripped = line.lstrip("#")
        cnt = len(line) - len(stripped)
        return cnt, stripped

    @classmethod
    def parse(cls, txt: str) -> Self:
        out = cls.root()
        stack = [ out ] #todo this can be avoided, I can simply traverse parents
        current = lambda: stack[-1]
        def pop():
            nonlocal stack
            stack = stack[:-1]
        lines = txt.split("\n")
        for line in lines:
            lvl, txt = cls._unprefix_header(line)
            if lvl:
                while current().level >= lvl:
                    pop()
                current().children.append(cls(txt, lvl, current()))
                stack.append(current().children[-1])
            else:
                current().content.append(txt)
        return out

    def find_root(self) -> Self:
        out = self
        while out.parent is not None:
            out = out.parent
        return out

    def tree(self) -> Iterable[Self]:
        yield self
        for child in self.children:
            yield from child.tree()

    @cached_property #todo you sure about this?
    def coordinates(self) -> tuple[int, ...]: #fixme this is really list; realign to tuple! it will cause a reindexing
        if self.parent is None:
            return []
        return self.parent.coordinates + [ self.parent.children.index(self) ]


    def deep_copy(self, *, include_own_content: bool = True, include_children_content: bool = True) -> Self:
        """
        :return: a detached copy
        """
        out = self.shallow_copy(include_own_content)
        for child in self.children:
            copied = child.deep_copy(
                #not a typo - include_own_content should only work when set by the consumer (user, developer)
                # otherwise, this method is called recursively, so include_children_content should be applied here
                include_own_content=include_children_content,
                include_children_content=include_children_content
            )
            copied.parent = out
            out.children.append(copied)
        return out

    def shallow_copy(self, include_content: bool = False) -> Self:
        """
        :return: a detached copy
        """
        return MarkdownChapter(self.title, self.level, None, list(self.content) if include_content else [])

    # @lru_cache #todo reenable; requires hashability
    def as_variant(self, variant: NoteChunkVariant | None, *, include_content: bool | None = None) -> Self:
        if variant is None:
            assert include_content is None
        include = lambda: {"include_content": include_content} if include_content is not None else {}
        match variant:
            case None | "simple": return self
            case "extracted": return self.as_extracted(**include())
            case "contextualized": return self.as_contextualized(**include())
            case _ as never: assert_never(never)

    def as_extracted(self, include_content: bool = False) -> Self:
        """
        Extracted chapter is the chapter and all is ancestors.

        Given
            # a
            A
            ## b
            B
            ### c
            C
            ## d
            D
            ### e
            E
        A, B, ... may contain subchapters, any formatting, etc; this only concerns the chapter structure.
        Then
            e.as_extracted(include_content=False)
        Is
            # a
            ## d
            ### e
            E
        While
            e.as_extracted(include_content=True)
        Is
            # a
            A
            ## d
            D
            ### e
            E
        Notice that `include_content` only impacts the ancestors content; self.content is always copied.

        :return: the extracter chapter; NOT the root of extracted chapter!
        """
        out = self.deep_copy(include_own_content=True, include_children_content=True)
        current = out
        parent = self.parent
        while parent is not None:
            copied_parent = parent.shallow_copy(include_content)
            current.parent = copied_parent
            copied_parent.children.append(current)
            current = copied_parent
            parent = parent.parent
        return out


    def as_contextualized(self, include_content: bool = False) -> Self:
        """
        Contextualized chapter is the chapter and all its previous siblings (without children) across all the ancestors.

        Given
            # a
            A
            ## b
            B
            ### c
            C
            ## d
            D
            ### e
            E
        A, B, ... may contain subchapters, any formatting, etc; this only concerns the chapter structure.
        Then
            c.as_contextualized(include_content=False)
        Is
            # a
            ## b
            ### c
            C
        While
            d.as_contextualized(include_content=False)
        Is
            # a
            ## b
            ## d
            D

        `include_content` works the same way as with `as_extracted`.
        For example
            d.as_contextualized(include_content=True)
        Is
            # a
            A
            ## b
            B
            ## d
            D
        Notice no c/C! It decides whether the content gets copied, but not the children.

        :return: the contextualized chapter; NOT the root of contextualized chapter!
        """
        current = self
        coordinates = [] # in reverse order; coordinates[0] is the index of self in self.parent.children; coordinates[-1] is the index in the root
        while current.parent is not None:
            parent = current.parent
            idx = parent.children.index(current)
            coordinates.append(idx)
            current = parent
        #coordinates = list(reversed(coordinates))
        # print("coordinates", self, coordinates)
        current = self
        idx = 0
        layers = [] # from root to parent, but not self
        while current.parent is not None:
            parent_copy = current.parent.shallow_copy(include_content)
            for i in range(coordinates[idx]):
                parent_copy.children.append(current.parent.children[i].shallow_copy(include_content))
            layers.append(parent_copy)
            idx += 1
            current = current.parent
        layers = list(reversed(layers))

        for parent, child in zip(layers, layers[1:]):
            parent.children.append(child)
            child.parent = parent

        # for child in self.parent.children:
        #     if child is self:
        #         break
        #     child_copy = child.shallow_copy(include_content)
        #     parent_copy.children.append(child_copy)
        #     child_copy.parent = parent_copy

        self_copy = self.deep_copy()
        if layers:
            parent_copy = layers[-1]
            parent_copy.children.append(self_copy)
            self_copy.parent = parent_copy
        return self_copy

#todo extract tests


def _set_parents(root: "MarkdownChapter") -> "MarkdownChapter":
    for c in root.children:
        c.parent = root
        _set_parents(c)
    return root

txt="""
# a
b
## c
## d
# e
### f
g
### h
i
## j
k

l
"""

md = _set_parents(MarkdownChapter.root(children=[
    MarkdownChapter(" a", 1, content=["b"], children=[
        MarkdownChapter(" c", 2),
        MarkdownChapter(" d", 2)
    ]),
    MarkdownChapter(" e", 1, content=[], children=[
        MarkdownChapter(" f", 3, content=["g"]),
        MarkdownChapter(" h", 3, content=["i"]),
        MarkdownChapter(" j", 2, content=["k", "", "l"]),
    ]),
]))

assert md.render() == txt.strip()
parsed = MarkdownChapter.parse(txt.strip())
assert parsed == md

parsed = MarkdownChapter.parse("""# a
A
## b
B
### c
C
## d
D
### e
E""")
a = parsed.children[0]
b = a.children[0]
c = b.children[0]
d = a.children[1]
e = d.children[0]

assert e.as_extracted(include_content=False).find_root().render() == """# a
## d
### e
E"""

assert e.as_extracted(include_content=True).find_root().render() == """# a
A
## d
D
### e
E"""

assert c.as_contextualized(include_content=False).find_root().render() == """# a
## b
### c
C"""

assert d.as_contextualized(include_content=False).find_root().render() == """# a
## b
## d
D
### e
E"""

assert d.as_contextualized(include_content=True).find_root().render() == """# a
A
## b
B
## d
D
### e
E"""


def break_to_entries(doc: Document[NoteMeta]) -> Iterable[Document[NoteChunkMeta]]:
    chapter = MarkdownChapter.parse(doc.content)
    already_seen = set()
    for subchapter in chapter.tree():
        simple = subchapter.render()
        if simple not in already_seen:
            yield Document(
                id=str(uuid4()),
                content=simple,
                metadata=NoteSimpleChunkMeta(
                    document_type = "chunk",
                    chunk_source_document_type = "note",
                    chunked_note_id = doc.id,
                    chunk_variant = "simple",
                    chunk_coordinates = subchapter.coordinates,
                    chunked_note_subject_path = doc.metadata.subject_path,
                    chunked_note_subject_type = doc.metadata.subject_type,
                )
            )
            already_seen.add(simple)
        for variant in DerivedChunkVariant.__args__:
            for inc_cont in [True, False]:
                try:
                    specialized = subchapter.as_variant(variant, include_content=inc_cont)
                    rendered = specialized.find_root().render()
                    if rendered not in already_seen:
                        yield Document(
                            id=str(uuid4()),
                            content=rendered,
                            metadata=NoteDerivedChunkMeta(
                                document_type="chunk",
                                chunk_source_document_type="note",
                                chunked_note_id=doc.id,
                                chunk_variant=variant,
                                chunk_coordinates=subchapter.coordinates,
                                chunk_include_previous_content=inc_cont,
                                chunked_note_subject_path=doc.metadata.subject_path,
                                chunked_note_subject_type=doc.metadata.subject_type
                            )
                        )
                        already_seen.add(rendered)
                except:
                    raise
