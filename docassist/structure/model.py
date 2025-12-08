from dataclasses import dataclass
from typing import NamedTuple, Self

class Chapter(NamedTuple):
    name: str
    content: list["Document"]
    preamble: str | None
    afterword: str | None

class Article(NamedTuple):
    name: str
    content: str

Document = Chapter | Article

class ChapterDefinition(NamedTuple):
    name: str
    preamble_description: str | None
    afterword_description: str | None
    content: list["DocumentDefinition"]

class ArticleDefinition(NamedTuple):
    name: str
    description: str

DocumentDefinition = ChapterDefinition | ArticleDefinition


class Let(dict): #this should be a dict
    def __init__(self, **values: "Descriptions"):
        self.update(values)

Descriptions = dict[str, str] | Let

def let(**values: Descriptions) -> Let:
    return Let(**values)


class Specification:
    #todo item in self
    def __rmatmul__(self, item: Descriptions) -> Self:
        return BindingScope[Self](item, self)


class Answer[T](NamedTuple):
    explanation: str
    value: T |  None

@dataclass(frozen=True)
class Variant[T: Specification](Specification):
    """
    Say, you want to count mocas, whatever they might be (I needed a mock word).
    If there is one, you want article called "Moca" about that single moca.
    If there are many, you want chapter "Mocas" explaining what mocas are in the preamble and having single article per each moca as
    content.
    If there is none, you skip the mocas altogether.
    You would do it as:
    DependsOnQuestion[DocumentDefinition](
        question="Is there no mocas, single moca or many mocas?",
        answers=[
            Answer(explanation="there are no mocas", value=None),
            # article describing that single moca, probably with templated title
            Answer(explanation="there is exactly one moca", value=ArticleDefinition(...)),
            # chapter with content of Article(..., depends_on_question=ForMany(subjects="each moca"))
            Answer(explanation="there is more than one moca", value=ChapterDefinition(...))
        ]
    )
    """
    question: str
    answers: list[Answer[T]]

def depends_on_question[T: Specification](question: str, answers: dict[str, T] | list[Answer[T]]) -> Variant[T]:
    if isinstance(answers, dict):
        answers = [ Answer(k, v) for k, v in answers.items() ]
    return Variant(
        question,
        answers
    )

def if_[T: Specification](question: str, yes: T | None = None, no: T | None = None) -> Variant[T]:
    return depends_on_question(question, [Answer("that is true", yes), Answer("that is false", no)])

@dataclass(frozen=True)
class BindingScope[T: Specification](Specification):
    variables: Descriptions
    content: T

#fluent API implemented in Specification as __contains__ + let()

@dataclass(frozen=True)
class Expansion[T: Specification](Specification):
    domain_description: str
    variables: Descriptions
    content_template: T

class expand_on(NamedTuple):
    domain_description: str

    def __and__[T](self, other: BindingScope[T]) -> Expansion[T]:
        return Expansion(self.domain_description, other.variables, other.content)

VariableValues = dict[str, str]

@dataclass(frozen=True)
class ChapterSpecification(Specification):
    name: str
    preamble_description: str | None
    afterword_description: str | None
    content: list["DocumentSpecification"]

def chapter(name: str, content: list["DocumentSpecification"],
            *, preamble_description: str | None = None, afterword_description: str | None = None) -> ChapterSpecification:
    return ChapterSpecification(name, preamble_description, afterword_description, content)

@dataclass(frozen=True)
class ArticleSpecification(Specification):
    name: str
    description: str

article = ArticleSpecification

DocumentSpecification = Specification | DocumentDefinition
