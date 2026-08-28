from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _as_float(
    name: str,
    default: float,
    minimum: float | None = None,
) -> float:
    value = float(os.getenv(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class RagV2Settings:
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10
    db_pool_timeout_seconds: float = 10.0
    db_statement_timeout_ms: int = 5000
    evren_base_url: str = "https://evren-llmapi.ssyz.org.tr/v1"
    evren_api_key: str = ""
    evren_embedding_model: str = "bge-m3-embed"
    evren_text_model: str = "llm-fast"
    evren_timeout_seconds: float = 180.0
    evren_connect_timeout_seconds: float = 10.0
    evren_max_retries: int = 2
    evren_max_output_tokens: int = 768
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "hititfinlex_rag_v2"
    qdrant_timeout_seconds: float = 30.0
    qdrant_max_retries: int = 2
    embedding_dimension: int = 1024
    dense_weight: float = 1.0
    lexical_weight: float = 0.5
    rrf_k: int = 60
    product_soft_boost: float = 0.001
    accepted_confidence: float = 0.80
    review_confidence: float = 0.65
    enforce_classification_policy: bool = False
    require_textual_product_confirmation: bool = False
    max_evidence: int = 12
    max_bank_evidence: int = 12
    candidate_multiplier: int = 8
    session_ttl_seconds: int = 2592000
    history_turns: int = 6
    history_max_chars: int = 16000
    conversation_summary_max_chars: int = 32000
    transcript_turns: int = 100
    route_with_llm: bool = True
    reranker_enabled: bool = False
    reranker_validated: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("accepted_confidence", self.accepted_confidence),
            ("review_confidence", self.review_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.review_confidence > self.accepted_confidence:
            raise ValueError(
                "review_confidence cannot exceed accepted_confidence"
            )

    @classmethod
    def from_env(cls) -> "RagV2Settings":
        required = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError(
                "Missing RAG V2 environment variables: " + ", ".join(missing)
            )

        settings = cls(
            db_host=os.environ["DB_HOST"],
            db_port=_as_int("DB_PORT", 5432),
            db_name=os.environ["DB_NAME"],
            db_user=os.environ["DB_USER"],
            db_password=os.environ["DB_PASSWORD"],
            db_pool_min_size=_as_int("RAG_V2_DB_POOL_MIN_SIZE", 2),
            db_pool_max_size=_as_int("RAG_V2_DB_POOL_MAX_SIZE", 10),
            db_pool_timeout_seconds=_as_float(
                "RAG_V2_DB_POOL_TIMEOUT_SECONDS", 10.0, 0.1
            ),
            db_statement_timeout_ms=_as_int(
                "RAG_V2_DB_STATEMENT_TIMEOUT_MS", 5000, 100
            ),
            evren_base_url=os.getenv(
                "EVREN_BASE_URL", "https://evren-llmapi.ssyz.org.tr/v1"
            ).rstrip("/"),
            evren_api_key=os.getenv("EVREN_API_KEY", ""),
            evren_embedding_model=os.getenv(
                "EVREN_EMBEDDING_MODEL", "bge-m3-embed"
            ),
            evren_text_model=os.getenv("EVREN_TEXT_MODEL", "llm-fast"),
            evren_timeout_seconds=_as_float(
                "EVREN_TIMEOUT_SECONDS", 180.0, 1.0
            ),
            evren_connect_timeout_seconds=_as_float(
                "EVREN_CONNECT_TIMEOUT_SECONDS", 10.0, 0.1
            ),
            evren_max_retries=_as_int("EVREN_MAX_RETRIES", 2, 0),
            evren_max_output_tokens=_as_int(
                "EVREN_MAX_OUTPUT_TOKENS", 768, 64
            ),
            qdrant_url=os.getenv("QDRANT_URL", "").rstrip("/"),
            qdrant_api_key=(
                os.getenv("QDRANT_API_KEY")
                or os.getenv("EVREN_QDRANT_KEY")
                or os.getenv("QDRANT_KEY")
                or ""
            ),
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION", "hititfinlex_rag_v2"
            ),
            qdrant_timeout_seconds=_as_float(
                "QDRANT_TIMEOUT_SECONDS", 30.0, 1.0
            ),
            qdrant_max_retries=_as_int("QDRANT_MAX_RETRIES", 2, 0),
            embedding_dimension=_as_int("EVREN_EMBEDDING_DIMENSION", 1024),
            dense_weight=_as_float("RAG_V2_DENSE_WEIGHT", 1.0, 0.0),
            lexical_weight=_as_float("RAG_V2_LEXICAL_WEIGHT", 0.5, 0.0),
            rrf_k=_as_int("RAG_V2_RRF_K", 60),
            product_soft_boost=_as_float(
                "RAG_V2_PRODUCT_SOFT_BOOST", 0.001, 0.0
            ),
            accepted_confidence=_as_float(
                "RAG_V2_ACCEPTED_CONFIDENCE", 0.80, 0.0
            ),
            review_confidence=_as_float(
                "RAG_V2_REVIEW_CONFIDENCE", 0.65, 0.0
            ),
            enforce_classification_policy=_as_bool(
                "RAG_V2_ENFORCE_CLASSIFICATION_POLICY", False
            ),
            require_textual_product_confirmation=_as_bool(
                "RAG_V2_REQUIRE_TEXTUAL_PRODUCT_CONFIRMATION", False
            ),
            max_evidence=_as_int("RAG_V2_MAX_EVIDENCE", 12),
            max_bank_evidence=_as_int("RAG_V2_MAX_BANK_EVIDENCE", 12),
            candidate_multiplier=_as_int(
                "RAG_V2_CANDIDATE_MULTIPLIER", 8
            ),
            session_ttl_seconds=_as_int("RAG_V2_SESSION_TTL_SECONDS", 2592000),
            history_turns=_as_int("RAG_V2_HISTORY_TURNS", 6),
            history_max_chars=_as_int("RAG_V2_HISTORY_MAX_CHARS", 16000, 1000),
            conversation_summary_max_chars=_as_int(
                "RAG_V2_CONVERSATION_SUMMARY_MAX_CHARS", 32000, 1000
            ),
            transcript_turns=_as_int("RAG_V2_TRANSCRIPT_TURNS", 100),
            route_with_llm=_as_bool("RAG_V2_ROUTE_WITH_LLM", True),
            reranker_enabled=_as_bool("RAG_V2_RERANKER_ENABLED", False),
            reranker_validated=_as_bool("RAG_V2_RERANKER_VALIDATED", False),
        )
        if settings.review_confidence > settings.accepted_confidence:
            raise ValueError(
                "RAG_V2_REVIEW_CONFIDENCE cannot exceed "
                "RAG_V2_ACCEPTED_CONFIDENCE"
            )
        if settings.db_pool_min_size > settings.db_pool_max_size:
            raise ValueError(
                "RAG_V2_DB_POOL_MIN_SIZE cannot exceed "
                "RAG_V2_DB_POOL_MAX_SIZE"
            )
        return settings

    @property
    def evren_ready(self) -> bool:
        return bool(self.evren_base_url and self.evren_api_key)

    @property
    def qdrant_ready(self) -> bool:
        return bool(self.qdrant_url and self.qdrant_api_key)
