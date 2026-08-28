from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


Intent = Literal[
    "lookup",
    "compare",
    "list",
    "calculate",
    "historical",
    "clarification",
    "chat",
]
Scope = Literal["current", "historical", "all"]
ChatStatus = Literal[
    "verified",
    "rejected",
    "insufficient_evidence",
    "needs_clarification",
    "conversational",
]


class OfferReference(BaseModel):
    offer_id: str
    bank: str
    product_types: list[str] = Field(default_factory=list)
    document_id: str | None = None
    rank: int = Field(ge=1)


class QueryRoute(BaseModel):
    standalone_query: str = ""
    intent: Intent = "lookup"
    banks: list[str] = Field(default_factory=list)
    product_types: list[str] = Field(default_factory=list)
    field_types: list[str] = Field(default_factory=list)
    scope: Scope = "current"
    year: int | None = Field(default=None, ge=1900, le=2100)
    date_from: date | None = None
    date_to: date | None = None
    offer_ids: list[str] = Field(default_factory=list)
    inherited_fields: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None

    @field_validator(
        "banks", "product_types", "field_types", "offer_ids", "inherited_fields"
    )
    @classmethod
    def unique_strings(cls, values: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = str(value).strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                output.append(cleaned)
        return output


class SessionState(BaseModel):
    active_banks: list[str] = Field(default_factory=list)
    broad_bank_context: bool = False
    active_products: list[str] = Field(default_factory=list)
    active_scope: Scope = "current"
    active_year: int | None = None
    active_date_from: date | None = None
    active_date_to: date | None = None
    active_offer_ids: list[str] = Field(default_factory=list)
    ranked_offers: list[OfferReference] = Field(default_factory=list)
    last_intent: Intent | None = None
    last_field_types: list[str] = Field(default_factory=list)
    last_source_ids: list[str] = Field(default_factory=list)
    last_document_ids: list[str] = Field(default_factory=list)
    last_standalone_query: str | None = None
    conversation_summary: str | None = None
    conversation_turn_count: int = Field(default=0, ge=0)


class StructuredFact(BaseModel):
    fact_type: str
    fact_text: str
    normalized_value: dict[str, Any] | None = None
    evidence_text: str
    confidence: float = Field(ge=0.0, le=1.0)


class SearchRecord(BaseModel):
    chunk_id: str
    offer_id: str
    document_id: str
    bank_key: str
    bank_name: str
    primary_product: str | None = None
    product_types: list[str] = Field(default_factory=list)
    page_title: str | None = None
    section_heading: str | None = None
    source_url: str | None = None
    scope: Scope
    effective_date: date | None = None
    campaign_start: date | None = None
    campaign_end: date | None = None
    content: str
    facts: list[StructuredFact] = Field(default_factory=list)
    classification_confidence: float = Field(ge=0.0, le=1.0)
    classification_status: Literal["accepted", "review", "required", "verified"]
    classification_conflict: bool = False
    product_scores: dict[str, float] = Field(default_factory=dict)
    dense_score: float | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rrf_score: float = 0.0
    product_boost: float = 0.0


class Evidence(BaseModel):
    source_id: str
    chunk_id: str
    offer_id: str
    document_id: str
    bank_name: str
    primary_product: str | None = None
    product_types: list[str] = Field(default_factory=list)
    page_title: str | None = None
    section_heading: str | None = None
    source_url: str | None = None
    scope: Scope
    effective_date: date | None = None
    campaign_start: date | None = None
    campaign_end: date | None = None
    content: str
    facts: list[StructuredFact] = Field(default_factory=list)
    classification_confidence: float
    classification_status: str
    classification_conflict: bool = False
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rrf_score: float


class RagV2ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, min_length=32, max_length=160)
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=12, ge=1, le=50)
    use_reranker: bool = False
    scope: Scope | None = None
    date_from: date | None = None
    date_to: date | None = None
    product_types: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if len(cleaned) < 2:
            raise ValueError("query must contain visible characters")
        return cleaned

    @field_validator("product_types")
    @classmethod
    def clean_product_types(cls, values: list[str]) -> list[str]:
        return list(
            dict.fromkeys(
                item.strip().upper()
                for item in values
                if isinstance(item, str) and item.strip()
            )
        )

    @model_validator(mode="after")
    def validate_date_order(self):
        if (
            self.date_from is not None
            and self.date_to is not None
            and self.date_from > self.date_to
        ):
            raise ValueError("date_from cannot be after date_to")
        return self


class RagV2ChatResponse(BaseModel):
    session_id: str
    query: str
    standalone_query: str
    answer: str
    status: ChatStatus
    inherited_context: dict[str, Any] = Field(default_factory=dict)
    route: QueryRoute
    evidence: list[Evidence] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    expires_at: datetime
    state: SessionState


class SessionClearResponse(BaseModel):
    session_id: str
    cleared: bool
    expires_at: datetime
    state: SessionState


class SessionTranscriptResponse(BaseModel):
    session_id: str
    expires_at: datetime
    messages: list[RagV2ChatResponse] = Field(default_factory=list)


class SessionDeleteResponse(BaseModel):
    deleted: bool
