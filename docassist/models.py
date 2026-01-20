from collections.abc import Callable
from decimal import Decimal
from enum import IntEnum
from typing import Literal, Self, Protocol

from genai_prices import calc_price, Usage
from genai_prices.types import ModelInfo, ClauseEquals, ClauseOr, ModelPrice
from pydantic import BaseModel



class Level(IntEnum):
    """
    Ordinal capability level.

    Levels are strictly ordered and compared numerically.
    Dispatcher semantics assume: model_level >= required_level.
    """

    NONE = 0
    """Capability is absent or explicitly unsupported.

    The model either cannot perform this function at all
    or is unreliable enough that it must be treated as unavailable.
    """

    BASIC = 1
    """Capability exists in a limited or fragile form.

    Works for simple cases or happy paths, but may fail under:
    - ambiguity
    - longer contexts
    - strict constraints
    - adversarial inputs

    Suitable only when failure is acceptable or downstream is robust.
    """

    RELIABLE = 2
    """Capability works consistently for intended use cases.

    The model usually:
    - follows instructions
    - respects constraints
    - recovers from minor ambiguity

    This is the default target level for production agents.
    """

    STRONG = 3
    """Capability is robust under complexity and edge cases.

    The model:
    - handles ambiguity well
    - maintains discipline under pressure
    - degrades gracefully instead of failing catastrophically

    Required for solver agents and high-stakes orchestration.
    """


class CapabilityVector(BaseModel):
    """
    Vector of orthogonal model capabilities.

    Each field represents an independent capability axis.
    Capability vectors are compared component-wise when selecting models.
    """

    model_config = {
        "frozen": True,
        "extra": "forbid",
    }

    output_formatting: Level
    """Ability to produce high-quality human-facing formatted text.

    Includes:
    - Markdown structure and consistency
    - headings, lists, emphasis
    - readable prose layout

    Does NOT include machine-parseable structure guarantees.
    Primarily relevant for writer-like agents.
    """

    structured_output: Level
    """Ability to emit machine-consumable structured outputs.

    Includes:
    - valid JSON / XML
    - schema adherence
    - correct use of `final_result`-style tools
    - avoiding extraneous text

    This is the core capability for doer and solver agents.
    """

    tool_use: Level
    """Ability to call tools correctly and at all.

    NONE:
        Model cannot call tools.
    BASIC:
        Can call tools but may misuse or miss them.
    RELIABLE:
        Selects appropriate tools and supplies correct arguments.
    STRONG:
        Uses tools strategically and efficiently in multi-step workflows.
    """

    tool_discipline: Level
    """Ability to obey tool-related constraints and contracts.

    Includes:
    - using 'I don't know' or empty-result tools instead of hallucinating
    - not emitting text when a tool-only response is required
    - respecting tool semantics under uncertainty

    This is intentionally separate from tool_use.
    """

    reasoning: Level
    """Ability to perform non-trivial inference and transformation.

    Covers:
    - multi-step reasoning
    - synthesis across inputs
    - resolving implicit constraints

    Does NOT include fact acquisition (see research).
    """

    research: Level
    """Ability to acquire missing information via tools.

    NONE:
        Operates only on provided input.
    BASIC:
        Can perform simple lookups.
    RELIABLE:
        Can conduct directed research and integrate results.
    STRONG:
        Can plan, execute, and verify multi-step research processes.

    Distinguishes doers from solvers.
    """

    epistemic_modesty: Level
    """Ability to acknowledge uncertainty or absence of an answer.

    Includes:
    - returning empty / null results when appropriate
    - explicitly invoking 'I don't know' mechanisms
    - avoiding forced answers under underspecification

    Writers typically require LOW or NONE here by design.
    """

    hallucination_resistance: Level
    """Resistance to fabricating unsupported facts or details.

    Measures how well the model:
    - avoids confabulation
    - stays grounded in available information
    - prefers abstention over invention

    Especially important for RAG pipelines and evaluators.
    """

    def satisfy(self, reqs: "CapabilityRequirements") -> bool:
        for field in self.model_fields:
            if getattr(self, field) < getattr(reqs, field):
                return False
        return True

    @classmethod
    def defaults(cls, default_level: Level) -> Self:
        return cls(**{k: default_level for k in CAPABILITY_NAMES})

    @classmethod
    def embeddings_only(cls) -> Self:
        return cls.defaults(Level.NONE)

    def override(self, **overrides: dict[str, Level]) -> Self: #dict key is Literal[*CAPABILITY_NAMES], but that's a bitch to express
        dumped = self.model_dump(mode="python")
        dumped.update(overrides)
        return type(self)(**dumped)

CAPABILITY_NAMES = list(CapabilityVector.model_fields.keys())
if "model_config" in CAPABILITY_NAMES:
    CAPABILITY_NAMES.remove("model_config")
CAPABILITY_NAMES = tuple(CAPABILITY_NAMES)

class CapabilityRequirements(CapabilityVector):
    output_formatting: Level = Level.NONE
    structured_output: Level = Level.NONE
    tool_use: Level = Level.NONE
    tool_discipline: Level = Level.NONE
    reasoning: Level = Level.NONE
    research: Level = Level.NONE
    epistemic_modesty: Level = Level.NONE
    hallucination_resistance: Level = Level.NONE

#this duplicates the call to broker constructor, but allows us to have model name enumeration
_KNOWN_MODELS = tuple([
    'deepseek/deepseek-r1-0528',
    # 'deepseek/deepseek-v3.2',
    'openai/gpt-oss-120b',
    'openai/gpt-oss-20b',
    'qwen/qwen3-32b',
    'text-embedding-3-small',
    "minimax/minimax-m1",
    "minimax/minimax-m2",
    "minimax/minimax-m2.1"
])

ModelName = Literal[*_KNOWN_MODELS]

class TokenInference(BaseModel):
    model_config = {
        "frozen": True,
        "extra": "forbid",
    }
    input: int
    #fixme missing reasoning and tool calls
    output: int
    sampled: bool

ModelObservation = TokenInference

class Tokens(BaseModel):
    model_config = {
        "frozen": True,
        "extra": "forbid",
    }

    input: int
    output: int

    @classmethod
    def zero(cls) -> Self:
        return cls(input=0, output=0)

    def __add__(self, other: Self) -> Self:
        return Tokens(input=self.input + other.input, output=self.output + other.output)

class TokenUsage(BaseModel):
    model_config = {
        "frozen": True,
        "extra": "forbid",
    }

    inferred: Tokens
    sampled: Tokens

    @property
    def total(self) -> Tokens:
        return self.inferred + self.sampled

    def __add__(self, other: Self) -> Self:
        return TokenUsage(inferred=self.inferred + other.inferred, sampled=self.sampled + other.sampled)

    @classmethod
    def from_observed(cls, inference: TokenInference) -> Self:
        t = Tokens(input=inference.input, output=inference.output)
        if inference.sampled:
            return cls(inferred=Tokens.zero(), sampled=t)
        return cls(inferred=t, sampled=Tokens.zero())

    @classmethod
    def zero(cls) -> Self:
        return cls(inferred=Tokens.zero(), sampled=Tokens.zero())


class ModelProfile(Protocol):
    @property
    def name(self) -> ModelName: ...

    @property
    def capabilities(self) -> CapabilityVector: ...

    def observe(self, event: ModelObservation): ...

    def get_usage(self) -> TokenUsage: ...


DrawStrategy = Callable[[list[ModelProfile]], ModelProfile]

from genai_prices.data import providers
openrouter = [ x for x in providers if x.id == "openrouter" ][0]
openrouter.models.extend([
    ModelInfo(
        id="deepseek-v3.2",
        match=ClauseEquals(equals='deepseek/deepseek-v3.2'),
        #mock prices from llm-prices.com (snapshot on 18/01/26)
        prices=ModelPrice(input_mtok=Decimal("0.25"), output_mtok=Decimal("0.38"))
    ),
    ModelInfo(
        id="minimax-m2",
        match=ClauseEquals(equals='minimax/minimax-m2'),
        prices=ModelPrice(input_mtok=Decimal("0.2"), output_mtok=Decimal("1"))
    ),
    ModelInfo(
        id="minimax-m2.1",
        match=ClauseEquals(equals='minimax/minimax-m2.1'),
        prices=ModelPrice(input_mtok=Decimal("0.27"), output_mtok=Decimal("1.12"))
    )
])

def cost_of(input_tokens: int, output_tokens: int, model_name: ModelName) -> float:
    try:
        return calc_price(
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            model_ref=model_name,
            provider_id="openrouter"
        ).total_price
    except LookupError:
        try:
            return calc_price(
                usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
                # we're using stuff like openai/gpt3 as model_name, while pricing library omits the prefix and uses gpt3
                model_ref=model_name.partition("/")[2],
                provider_id="openrouter"
            ).total_price
        except:
            raise

def cheapest_for_inference_of(*, input_tokens: int, output_tokens: int) -> DrawStrategy:
    def strategy(candidate_models: list[ModelProfile]) -> ModelProfile:
        return min(candidate_models, key=lambda profile: cost_of(input_tokens, output_tokens, profile.name))
    return strategy

DEFAULT_STRATEGY = cheapest_for_inference_of(input_tokens=2000, output_tokens=500)

class ModelProfileOverState(ModelProfile):
    def __init__(self, state: "ModelState"):
        self._state = state

    @property
    def name(self) -> ModelName:
        return self._state.name

    @property
    def capabilities(self) -> CapabilityVector:
        return self._state.capabilities

    def observe(self, event: ModelObservation):
        return self._state.observe(event)

    def get_usage(self) -> TokenUsage:
        return self._state.usage

class ModelState(BaseModel):
    name: str
    capabilities: CapabilityVector
    usage: TokenUsage = TokenUsage.zero()

    def observe(self, event: ModelObservation):
        match event:
            case TokenInference() as inference: self._observe_inferred_tokens(inference)
            case _ as never: assert_never(never)

    def _observe_inferred_tokens(self, inference: TokenInference):
        usage_delta = TokenUsage.from_observed(inference)
        self.usage += usage_delta

    def as_profile(self) -> ModelProfile:
        return ModelProfileOverState(self)

class ModelBroker:
    def __init__(self, models: dict[ModelName, CapabilityVector]):
        self._state: dict[ModelName, ModelState] = {}
        for model_name, capabilities in models.items():
            self._register_model(model_name, capabilities)
        expected = set(_KNOWN_MODELS)
        actual = set(self._state.keys())
        missing = expected.difference(actual)
        unexpected = actual.difference(expected)
        assert not missing, str(missing)
        assert not unexpected, str(unexpected)

    def _register_model(self, model_name: ModelName, capabilities: CapabilityVector):
        self._state[model_name] = ModelState(name=model_name, capabilities=capabilities)

    @property
    def model_names(self) -> list[ModelName]:
        return list(_KNOWN_MODELS)

    @property
    def model_profiles(self) -> list[ModelProfile]:
        return [
            self.get_model_profile(name)
            for name in self.model_names
        ]

    def get_model_profile(self, name: str) -> ModelProfile:
        return self._state[name].as_profile()

    def pick_model_profile(self, requirements: CapabilityRequirements, draw_strategy: DrawStrategy = DEFAULT_STRATEGY) -> ModelProfile | None:
        satisfying = [model for model in self.model_profiles if model.capabilities.satisfy(requirements)]

        if not satisfying:
            return None
        if len(satisfying) == 1:
            return satisfying[0]
        return draw_strategy(satisfying)


    # simplified for now; f we add latency, value type should be higher-level report containing usage and latency
    def report(self) -> dict[ModelName | Literal["total"], TokenUsage]:
        total = TokenUsage.zero()
        out = {

        }
        for state in self._state.values():
            out[state.name] = state.usage
            total += state.usage
        out["total"] = total
        return out

