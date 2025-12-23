from pydantic import BaseModel

# def Explained[T]()


class DomainInterpretationInput(BaseModel):
    domain_description: str
    # requested_count: int | None = None

class InterpretedDomain(BaseModel):
    target_kind: str
    qualifiers: list[str]
    cardinality: int | None
    notes: str | None


class CandidateHypothesis(BaseModel):
    description: str
    retrieval_hints: list[str]
    explanation: str


class AnchoredEntity(BaseModel):
    entity_id: str
    label: str
    evidence_documents: list[str]
    confidence: float
    explanation: str

class EntityAnchorInput(BaseModel):
    domain: InterpretedDomain
    candidates: list[CandidateHypothesis]

class EntityAnchorOutput(BaseModel):
    entities: list[AnchoredEntity]

class SchemaVariable(BaseModel):
    name: str
    description: str

class SchemaProjectorInput(BaseModel):
    entity: AnchoredEntity
    variables: list[SchemaVariable]

class SchemaProjectedItem(BaseModel):
    entity_id: str
    values: dict[str, str]
    evidence: list[str]
    explanation: str
