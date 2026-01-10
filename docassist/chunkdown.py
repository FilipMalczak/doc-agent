from dataclasses import field, dataclass
from functools import cached_property
from typing import Self, Iterable, assert_never, Literal

from docassist.index.document import Document

SIMPLE_VARIANT = "self.title+self.content"
SUBTREE_VARIANT = "self.title+self.content+self.children[*]"
SHALLOW_SUBTREE_VARIANT = "self.title+self.content+self.children[*].title"
WITH_ANCESTOR_TITLES_VARIANT = "ancestors[*].title+self.title+self.content"

VARIANTS = [ SIMPLE_VARIANT, SUBTREE_VARIANT, SHALLOW_SUBTREE_VARIANT, WITH_ANCESTOR_TITLES_VARIANT ]

SimpleVariant = Literal[SIMPLE_VARIANT]
SubtreeVariant = Literal[SUBTREE_VARIANT]
ShallowSubtreeVariant = Literal[SHALLOW_SUBTREE_VARIANT]
WithAncestorTitlesVariant = Literal[WITH_ANCESTOR_TITLES_VARIANT]

MarkdownChapterVariant = SimpleVariant | SubtreeVariant | ShallowSubtreeVariant | WithAncestorTitlesVariant

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

    def shallow_copy(self, include_content: bool = True) -> Self:
        """
        :return: a detached copy
        """
        return MarkdownChapter(self.title, self.level, None, list(self.content) if include_content else [])

    # @lru_cache #todo reenable; requires hashability
    def as_variant(self, variant: MarkdownChapterVariant = SIMPLE_VARIANT) -> Self:
        if variant == SIMPLE_VARIANT: return self.as_simple()
        elif variant == SUBTREE_VARIANT: return self.deep_copy()
        elif variant == SHALLOW_SUBTREE_VARIANT: return self.as_shallow_subtree()
        elif variant == WITH_ANCESTOR_TITLES_VARIANT: return self.as_with_ancestor_titles()
        else: assert_never(variant)

    def as_simple(self) -> Self:
        return MarkdownChapter(self.title, self.level, None, list(self.content), [])

    def as_shallow_subtree(self) -> Self:
        out = MarkdownChapter(self.title, self.level, None, list(self.content))
        for c in self.children:
            child_copy = MarkdownChapter(c.title, c.level, out, [], [])
            out.children.append(child_copy)
        return out

    def as_with_ancestor_titles(self) -> Self:
        out = self.shallow_copy()
        x = self.parent
        while x is not None:
            new_parent = x.shallow_copy(include_content=False)
            out.parent = new_parent
            new_parent.children.append(out)
            out = new_parent
            x = x.parent
        return out

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

assert e.as_variant(SIMPLE_VARIANT).render() == """### e
E"""

assert d.as_variant(SIMPLE_VARIANT).render() == """## d
D"""

assert a.as_variant(SUBTREE_VARIANT).render() == a.render()

assert d.as_variant(SUBTREE_VARIANT).render() == """## d
D
### e
E"""

assert e.as_variant(SUBTREE_VARIANT).render() == """### e
E"""

assert a.as_variant(SHALLOW_SUBTREE_VARIANT).render() == """# a
A
## b
## d"""

assert d.as_variant(SHALLOW_SUBTREE_VARIANT).render() == """## d
D
### e"""

assert e.as_variant(SHALLOW_SUBTREE_VARIANT).render() == """### e
E"""


assert a.as_variant(WITH_ANCESTOR_TITLES_VARIANT).render() == """# a
A"""

assert b.as_variant(WITH_ANCESTOR_TITLES_VARIANT).render() == """# a
## b
B"""

assert c.as_variant(WITH_ANCESTOR_TITLES_VARIANT).render() == """# a
## b
### c
C"""

assert e.as_variant(WITH_ANCESTOR_TITLES_VARIANT).render() == """# a
## d
### e
E"""


#these are Documents to avoid cyclic import; they could be typed with stuff from preindexing_graph
def break_to_entries(doc: Document) -> Iterable[Document]:
    chapter = MarkdownChapter.parse(doc.content)
    already_seen = set()
    for subchapter in chapter.tree():
        for variant in VARIANTS:
            rendered = subchapter.as_variant(variant).render()
            if rendered not in already_seen:
                yield doc.derive_note_chapter(rendered, subchapter.coordinates, variant)
