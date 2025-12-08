from logging import getLogger
from typing import assert_never

from logfire import instrument

from docassist.agents.rag.tool import SearchIndexTool
from docassist.index.protocols import DocumentIndex
from docassist.sampling.protocols import SamplingController
from docassist.structure.materialize.binding.rephrase_variable_description_prompt import rephrase_variable_description
from docassist.structure.materialize.binding.resolve_variable_value_prompt import resolve_variable_value
from docassist.structure.materialize.expansion.expand_domain_prompt import expand_domain
from docassist.structure.materialize.models import MaterializationState, RephraseDescriptionInput, \
    VariableSpecification, RephraseDescriptionOutput, ResolveVariableInput, VariableValuation, ResolveVariableOutput, \
    ExpandAnswersInput, IndexedAnswer, ExpandAnswersOutput, ChooseTheAnswerInput, ChooseTheAnswerOutput, \
    ExpandOnDomainInput, ExpandOnDomainOutput
from docassist.structure.materialize.variant.choose_the_answer_prompt import choose_the_answer
from docassist.structure.materialize.variant.expand_the_answers_prompt import expand_the_answers
from docassist.structure.model import DocumentSpecification, DocumentDefinition, VariableValues, ChapterSpecification, \
    ChapterDefinition, ArticleDefinition, ArticleSpecification, BindingScope, Variant, Expansion

logger = getLogger(__name__)


def _shadow(current: VariableValues, new: VariableValues) -> VariableValues:
    out = dict(current)
    out.update(new)
    return out

#todo typing
def _flatten(i):
    out = []
    for x in i:
        out.extend(x)
    return out


class Materializer:
    def __init__(self, index: DocumentIndex, controller: SamplingController, search_tool: SearchIndexTool):
        self.index: DocumentIndex = index
        self.controller: SamplingController = controller
        self.search_tool: SearchIndexTool = search_tool

    @instrument()
    def _materialize(self, specification: DocumentSpecification,
                     state: MaterializationState
                     ) -> list[DocumentDefinition]:
        match specification:
            case None:
                return []
            case ChapterDefinition() | ArticleDefinition() as definition:
                return [definition]
            case ChapterSpecification() as c:
                return self._materialize_chapter(c, state)
            case ArticleSpecification() as a:
                return self._materialize_article(a, state)
            case BindingScope() as scope:
                return self._materialize_scope(scope, state)
            case Variant() as v:
                return self._materialize_variant(v, state)
            case Expansion() as e:
                return self._materialize_expansion(e, state)
            case _ as never:
                assert_never(never)

    @instrument()
    def _materialize_chapter(self, c: ChapterSpecification,
                             state: MaterializationState) -> list[ChapterDefinition]:
        new_name = state.format_text_template(c.name)
        new_preamble = state.format_text_template(c.preamble_description)
        new_afterword = state.format_text_template(c.afterword_description)
        new_state = state.add_ancestor(new_name, new_preamble)

        return [
            ChapterDefinition(
                new_name,
                new_preamble,
                new_afterword,
                _flatten(
                    self._materialize(spec, new_state)
                    for spec in c.content
                )
            )
        ]

    @instrument()
    def _materialize_article(self, a: ArticleSpecification,
                             state: MaterializationState) -> list[ArticleDefinition]:
            return [
                ArticleDefinition(
                    state.format_text_template(a.name),
                    state.format_text_template(a.description)
                )
            ]


    @instrument
    def _resolve_variable(self, name, desc, state: MaterializationState, llmio: LLMIO) -> VariableValuation: #todo return type
        var_spec = VariableSpecification(name=name, description=desc) #todo this should happen on call site
        raw_rephrasings = self.model.feed(
            rephrase_variable_description(
                RephraseDescriptionInput(
                    desired_rewrite_count=10,
                    variable_specification=var_spec,
                    specification_context=state.ancestors,
                    resolved_variables=state.variable_values
                ),
                self.llmio
            )
        ).content
        rephrasings = self.llmio.lax_parse(raw_rephrasings, RephraseDescriptionOutput).rewrites
        task = variable_evaluation(name, desc)
        docs = self.reranker.rerank_results(task, self.index.find(rephrasings+[name], n=10).all())
        logger.info(f"RAG result (task: {task}:")
        for i, x in enumerate(docs):
            logger.info(f"DOCUMENT {i+1}/{len(docs)}:")
            logger.info(x.entry.id+" //// "+str(x.score))
            logger.info(x.entry.txt)
            logger.info("-"*80)
        response = self.model.feed(
            resolve_variable_value(
                ResolveVariableInput(
                    variable_specification=var_spec,
                    resources=DocumentChunk.from_items(docs)
                ),
                self.llmio
            )
        ).content
        valuation = llmio.lax_parse(response, ResolveVariableOutput)
        logger.info(f"Variable {name} -> {valuation.value}")
        return valuation

    @instrument()
    def _materialize_scope[T](self, s: BindingScope[T],
                             state: MaterializationState) -> list[DocumentDefinition]:
        new_state = state
        for var_name, var_desc in s.variables.items():
            explained_value = self._resolve_variable(var_name, var_desc, state, self.llmio)
            new_state = (new_state
                     .set_variable(var_name, explained_value)
                        #todo add "in context of <ancestors> the value..."; at that point - probably automatize it
                     .add_fact(f"The value of variable is '{explained_value.value}")
                     .add_fact(f"The value of variable '{var_name}' defined as '{var_desc}' is '{explained_value.value}")
                     .add_fact(f"The value of variable '{var_name}' defined as '{var_desc}' is '{explained_value.value} with explanation: {explained_value.explanation}'"))
        return self._materialize(s.content, new_state)

    @instrument()
    def _materialize_variant[T](self, v: Variant[T],
                             state: MaterializationState) -> list[DocumentDefinition]:
        logger.info(f"Gathering data to answer {v.question}")
        with self.controller.defer_until_success():
            response = self.model.feed(expand_the_answers(ExpandAnswersInput(
                expansions_count=5,
                question=v.question,
                answers=[IndexedAnswer(index=idx, answer=a.explanation) for idx, a in enumerate(v.answers)]
            ), self.llmio))
            expansions = self.llmio.lax_parse(response.content, ExpandAnswersOutput)
            queries = expansions.expanded
            assert queries
        logger.info(f"Queries: {queries}")
        docs = self.index.find(queries, n=5).grouped()
        docs = self._rerank(closed_question(v.question, [a.explanation for a in v.answers]), docs)
        logger.info("RAG result:")
        for i, x in enumerate(docs):
            logger.info(f"DOCUMENT {i + 1}/{len(docs)}:")
            logger.info(x.entry.id + " //// " + str(x.score))
            logger.info(x.entry.txt)
            logger.info("-" * 80)
        logger.info(f"Figuring out: {v.question}")
        # fixme we're not using ancestors here
        response = self.model.feed(
            choose_the_answer(
                ChooseTheAnswerInput(
                    question=v.question,
                    answers=[IndexedAnswer(index=i+1, answer=a.explanation) for i, a in enumerate(v.answers)],
                    resources=DocumentChunk.from_items(
                        docs + [ #todo these should be added in reranking stage
                            ScoredItem(
                                entry=IndexedEntry.make(
                                    "memory:processing_notes",
                                    {},
                                    self.llmio.dump(Facts(facts=state.facts))
                                ),
                                score=1.0
                            )
                        ] if state.facts else []
                    )
                ),
                self.llmio
            )
        )
        choice = self.llmio.lax_parse(response.content, ChooseTheAnswerOutput)
        idx = int(choice.chosen_index) - 1 # we give answers with 1-based index, we want 0-based one here
        logger.info(f"LLM chose answer #{idx+1}: {v.answers[idx] if idx < len(v.answers) and idx > 0 else 'I dont know'}")
        logger.info(f"The explanation of the choice was: <<{choice.explanation}>>")
        assert idx < len(v.answers), "The LLM cannot reply with 'I dont know'"
        chosen = v.answers[idx]
        logger.info(f"Proceeding with {chosen}")
        #todo add fact 'when presented with question ... we decided ...[ because ...]'
        chosen_materialized = self._materialize(chosen.value, state)
        return chosen_materialized

    @instrument()
    def _materialize_expansion[T](self, exp: Expansion[T],
                                  state: MaterializationState) -> list[DocumentDefinition]:
        out = []
        # fixme we're not using ancestors here
        response = self.model.feed(
            expand_domain(
                input=ExpandOnDomainInput(
                    domain_description=exp.domain_description,
                    variables=exp.variables,
                    resources=self.rag.retrieve(
                        index=self.index,
                        task=expansion(exp.domain_description),
                        queries=[exp.domain_description]
                    )
                ),
                llmio=self.llmio
            )
        )
        result = self.llmio.lax_parse(response.content, ExpandOnDomainOutput).results
        logger.info("Examples:")
        for r in result:
            logger.info(pformat(r))
            new_state = state
            for k, v in r.values.items():
                new_state = (new_state
                             .set_variable(k, v)
                             .add_fact(f"The value of variable '{k}' defined as '{exp.variables[k]}' is '{v}'."))
            out.extend(self._materialize(exp.content_template, new_state))
        return out

    @instrument()
    def materialize_specification(self, specification: DocumentSpecification) -> list[DocumentDefinition]:
        return self._materialize(specification, state=MaterializationState(ancestors=[], variable_values={}, facts=[]))