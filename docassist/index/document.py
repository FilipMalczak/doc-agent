from typing import Literal, Annotated, TypedDict, Any, Self
from uuid import uuid4

from pydantic import BaseModel

from docassist.idgen import make_id
# from docassist.chunkdown import MarkdownChapterVariant
from docassist.preindexing.perspectives import PerspectivePointer
from docassist.subjects import repo_function

DocumentId = str
Text = str
Content = Text
Query = Text
Distance = float



#todo extract document_type

SourceDocumentType = Literal["source_file"]

NoteDocumentType = Literal["note"]
FactsDocumentType = Literal["facts"]

NoteChapterDocumentType = Literal["note_chapter"]
FactDocumentType = Literal["single_fact"]

DerivedDocumentType = NoteDocumentType | FactsDocumentType | NoteChapterDocumentType | FactDocumentType

TransientDocumentType = Literal["transient"]

def value_of(type_literal) -> str:
    out = type_literal.__args__
    assert len(out) == 1
    return out[0]

DocumentType = SourceDocumentType | TransientDocumentType | DerivedDocumentType

class Subject(BaseModel):
    id: str
    document_type: DocumentType

class Derivation(BaseModel):
    subject: Subject
    details: dict

class SourceFileDetails(TypedDict):
    repo_item_type: repo_function
    path: str
    language: str

class Document(BaseModel):
    # identity: list[Derivation]
    id: str
    document_type: DocumentType
    details: dict[str, Any]
    provenance: list[Derivation]
    content: str
    additional_details: dict[str, Any] | None

    # @property
    # def id(self) -> str:
    #     return self.identity[-1]["subject"]["id"]
    #
    # @property
    # def document_type(self) -> DocumentType:
    #     return self.identity[-1]["subject"]["document_type"]

    @property
    def generation(self) -> int:
        return len(self.identity) + 1


    def to_prompt_dict(self) -> dict[str, Any]:
        out = self.model_dump(mode="json")
        if out.get("additional_details", None):
            out["details"].update(out["additional_details"])
            del out["additional_details"]
        return out

    @classmethod
    def source_file(cls,
                    content: str, *,
                    path: str,
                    language: str,
                    repo_function: repo_function,
                    used_id: str | None = None
                    ) -> Self:
        # subject = Subject(id=used_id or str(uuid4()), document_type=value_of(SourceDocumentType))
        # subject = {"id": used_id or str(uuid4()), "document_type": value_of(SourceDocumentType)}
        # identity = [
        #     {"subject": subject, "details": {"path": path}}
        # ]
        doctype = value_of(SourceDocumentType)
        return cls(
            id=make_id(doctype, content),
            document_type=doctype,
            details={"path": path},
            provenance=[],
            content=content,
            additional_details={"language": language, "repo_function": repo_function}
        )
        # identity = [
        #     {"subject": subject, "details": {"path": path}}
        # ]
        # return Document(identity=identity, content=content, additional_details={"language": language, "repo_function": repo_function})
        #

    @classmethod
    def transient(cls, content: str, *, used_id: str | None = None) -> Self:
        # subject = {"id": str(uuid4()), "document_type": value_of(TransientDocumentType)}
        # identity = [
        #     {"subject": subject, "details": {}}
        # ]
        # return Document(identity=identity, content=content, additional_details=None)
        doctype = value_of(TransientDocumentType)
        return cls(
            id=make_id(doctype, content),
            document_type=doctype,
            details={},
            provenance=[],
            content=content,
            additional_details=None
        )


    def derive[Dets: TypedDict](self,
                                doc_type: DerivedDocumentType, content: str, details: Dets, *,
                                used_id: str | None = None, additional_details: dict[str, Any] | None = None
                                ) -> Self:
        # subject = {"id": used_id or str(uuid4()), "document_type": doc_type}
        # new_derivation = {"subject": subject, "details": details}
        # identity = self.identity + [ new_derivation ]
        # return Document(identity=identity, content=content, additional_details=additional_details)
        return Document(
            id = make_id(doc_type, content),
            document_type=doc_type,
            details=details,
            provenance=self.provenance + [
                # {
                #     "subject": {"id": self.id, "document_type": self.document_type},
                #     "details": self.details
                # }
                Derivation(
                    subject=Subject(id=self.id, document_type=self.document_type),
                    details=self.details
                )
            ],
            content=content,
            additional_details=additional_details
        )

    def derive_note(self, content: str, perspective: PerspectivePointer) -> Self:
        assert self.document_type == value_of(SourceDocumentType)
        return self.derive(
            doc_type=value_of(NoteDocumentType),
            content=content,
            details={
                "perspective": perspective.model_dump(mode="json")
            }
        )


    def derive_note_chapter(self, content: str, coordinates: tuple[int], variant) -> Self:
        """

        :param variant: docassist.chunkdown.MarkdownChapterVariant; simplified to avoid circular dependency
        """
        assert self.document_type == value_of(NoteDocumentType)
        return self.derive(
            doc_type=value_of(NoteChapterDocumentType),
            content=content,
            details={
                "chapter_coordinates": ",".join(map(str, coordinates)),
                "variant": variant
            }
        )

    def derive_facts(self, content: str, perspective: PerspectivePointer) -> Self:
        assert self.document_type == value_of(SourceDocumentType)
        return self.derive(
            doc_type=value_of(FactsDocumentType),
            content=content,
            details={
                "perspective": perspective.model_dump(mode="json")
            }
        )

    def derive_fact(self, content: str, index: int, explained: bool) -> Self:
        assert self.document_type == value_of(FactsDocumentType)
        return self.derive(
            doc_type=value_of(FactDocumentType),
            content=content,
            details={
                "fact_index": index,
                "include_explanation": explained
            }
        )
#
#
# class CommonMetadata(BaseModel):
#     document_type: DocumentType
#
# FileSubjectType = Literal["file"]
# DirSubjectType = Literal["directory"]
#
# NoteSubjectType = FileSubjectType | DirSubjectType
#
# class SourceMeta(CommonMetadata):
#     document_type: SourceDocumentType
#     type: FileSubjectType
#     repo_item_type: RepoItemType
#     path: str
#     language: str
#
#
# class CommonNoteMetadata(CommonMetadata):
#     document_type: NoteDocumentType
#     subject_path: str
#     subject_type: NoteSubjectType
#
#
# class FileNoteMeta(CommonNoteMetadata):
#     subject_id: str
#     subject_type: FileSubjectType
#     subject_document_type: SourceDocumentType
#     subject_repo_item_type: RepoItemType
#     subject_language: str
#
#
# class DirNoteMeta(CommonNoteMetadata):
#     subject_type: DirSubjectType
#
#
# NoteMeta = Annotated[FileNoteMeta | DirNoteMeta, Field(discriminator="subject_type")]
#
#
# SimpleChunkVariant = Literal["simple"]
# DerivedChunkVariant = Literal["extracted", "contextualized"]
# ExplainedChunkVariant = Literal["explained"]
#
# NoteChunkVariant = SimpleChunkVariant | DerivedChunkVariant
# FactsChunkVariant = SimpleChunkVariant | ExplainedChunkVariant
#
# ChunkVariant = NoteChunkVariant | FactsChunkVariant
#
#
# class CommonChunkMeta(CommonMetadata):
#     document_type: ChunkDocumentType
#     # do not add `chunk_source_document_id`
#     # add `chunked_<something>_id`, like `chunked_note_id`
#     chunk_source_document_type: DocumentType
#     chunk_variant: ChunkVariant
#     chunk_coordinates: tuple[int, ...]
#
#     def chunk_source_document_id(self) -> str:
#         raise NotImplemented("Implement in subclasses and delegate to appropriate field")
#
# class NoteSimpleChunkMeta(CommonChunkMeta):
#     chunk_source_document_type: NoteDocumentType
#     chunked_note_id: str
#     chunk_variant: SimpleChunkVariant
#     chunked_note_subject_path: str
#     chunked_note_subject_type: NoteSubjectType
#
#     def chunk_source_document_id(self) -> str:
#         return self.chunked_note_id
#
#
# class NoteDerivedChunkMeta(NoteSimpleChunkMeta):
#     chunk_variant: DerivedChunkVariant
#     chunk_include_previous_content: bool
#
#
# NoteChunkMeta = Annotated[NoteSimpleChunkMeta | NoteDerivedChunkMeta, Field(discriminator="chunk_variant")]
#
#
# class FactsMeta(CommonMetadata):
#     document_type: FactsDocumentType
#     subject_id: str
#     subject_path: str
#     subject_type: FileSubjectType
#     subject_document_type: SourceDocumentType
#     subject_repo_item_type: RepoItemType
#     subject_language: str
#
#
# class FactsChunkMeta(CommonChunkMeta):
#     chunk_source_document_type: FactsDocumentType
#     chunk_variant: FactsChunkVariant
#     chunked_facts_id: str
#     chunked_facts_subject_path: str
#     chunked_facts_subject_type: FileSubjectType
#
#     def chunk_source_document_id(self) -> str:
#         return self.chunked_facts_id
#
# ChunkMeta = Annotated[ NoteChunkMeta | FactsChunkMeta, Field(discriminator="chunk_source_document_type")]
#
# class TransientMeta(CommonMetadata):
#     document_type: TransientDocumentType
#
# AnyMetadata = Annotated[SourceMeta | NoteMeta | FactsMeta | ChunkMeta | TransientMeta, Field(discriminator="document_type")]
#
# class Document[Meta: AnyMetadata](BaseModel):
#     id: DocumentId
#     content: Content
#     metadata: Meta
