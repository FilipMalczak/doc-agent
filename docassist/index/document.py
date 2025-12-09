from typing import Literal, Annotated

from pydantic import BaseModel, Field

from docassist.subjects import RepoItemType

DocumentId = str
Text = str
Content = Text
Query = Text
Distance = float



#todo extract document_type

SourceDocumentType = Literal["source_file"]
NoteDocumentType = Literal["note"]
FactsDocumentType = Literal["facts"]
ChunkDocumentType = Literal["chunk"]

TransientDocumentType = Literal["transient"]

DocumentType = SourceDocumentType | NoteDocumentType | FactsDocumentType | ChunkDocumentType | TransientDocumentType

class CommonMetadata(BaseModel):
    document_type: DocumentType

FileSubjectType = Literal["file"]
DirSubjectType = Literal["directory"]

NoteSubjectType = FileSubjectType | DirSubjectType

class SourceMeta(CommonMetadata):
    document_type: SourceDocumentType
    type: FileSubjectType
    repo_item_type: RepoItemType
    path: str
    language: str


class CommonNoteMetadata(CommonMetadata):
    document_type: NoteDocumentType
    subject_path: str
    subject_type: NoteSubjectType


class FileNoteMeta(CommonNoteMetadata):
    subject_id: str
    subject_type: FileSubjectType
    subject_document_type: SourceDocumentType
    subject_repo_item_type: RepoItemType
    subject_language: str


class DirNoteMeta(CommonNoteMetadata):
    subject_type: DirSubjectType


NoteMeta = Annotated[FileNoteMeta | DirNoteMeta, Field(discriminator="subject_type")]


SimpleChunkVariant = Literal["simple"]
DerivedChunkVariant = Literal["extracted", "contextualized"]
ExplainedChunkVariant = Literal["explained"]

NoteChunkVariant = SimpleChunkVariant | DerivedChunkVariant
FactsChunkVariant = SimpleChunkVariant | ExplainedChunkVariant

ChunkVariant = NoteChunkVariant | FactsChunkVariant


class CommonChunkMeta(CommonMetadata):
    document_type: ChunkDocumentType
    # do not add `chunk_source_document_id`
    # add `chunked_<something>_id`, like `chunked_note_id`
    chunk_source_document_type: DocumentType
    chunk_variant: ChunkVariant
    chunk_coordinates: tuple[int, ...]

    def chunk_source_document_id(self) -> str:
        raise NotImplemented("Implement in subclasses and delegate to appropriate field")

class NoteSimpleChunkMeta(CommonChunkMeta):
    chunk_source_document_type: NoteDocumentType
    chunked_note_id: str
    chunk_variant: SimpleChunkVariant
    chunked_note_subject_path: str
    chunked_note_subject_type: NoteSubjectType

    def chunk_source_document_id(self) -> str:
        return self.chunked_note_id


class NoteDerivedChunkMeta(NoteSimpleChunkMeta):
    chunk_variant: DerivedChunkVariant
    chunk_include_previous_content: bool


NoteChunkMeta = Annotated[NoteSimpleChunkMeta | NoteDerivedChunkMeta, Field(discriminator="chunk_variant")]


class FactsMeta(CommonMetadata):
    document_type: FactsDocumentType
    subject_id: str
    subject_path: str
    subject_type: FileSubjectType
    subject_document_type: SourceDocumentType
    subject_repo_item_type: RepoItemType
    subject_language: str


class FactsChunkMeta(CommonChunkMeta):
    chunk_source_document_type: FactsDocumentType
    chunk_variant: FactsChunkVariant
    chunked_facts_id: str
    chunked_facts_subject_path: str
    chunked_facts_subject_type: FileSubjectType

    def chunk_source_document_id(self) -> str:
        return self.chunked_facts_id

ChunkMeta = Annotated[ NoteChunkMeta | FactsChunkMeta, Field(discriminator="chunk_source_document_type")]

AnyMetadata = Annotated[SourceMeta | NoteMeta | FactsMeta | ChunkMeta, Field(discriminator="document_type")]

class Document[Meta: AnyMetadata](BaseModel):
    id: DocumentId
    content: Content
    metadata: Meta
