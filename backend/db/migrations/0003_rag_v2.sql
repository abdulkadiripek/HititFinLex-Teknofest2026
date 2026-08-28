-- RAG V2 unified lexical index and server-side conversation persistence.
-- Target: PostgreSQL 18. This migration is safe to execute more than once.

CREATE EXTENSION IF NOT EXISTS unaccent WITH SCHEMA public;

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id CHAR(64) PRIMARY KEY,
    offer_id CHAR(64) NOT NULL,
    scope VARCHAR(10) NOT NULL
        CHECK (scope IN ('current', 'historical')),
    document_id VARCHAR(96) NOT NULL,
    current_source_id BIGINT REFERENCES documents(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    historical_source_id BIGINT REFERENCES historical_documents(id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content_hash CHAR(64) NOT NULL,
    bank_key VARCHAR(100) NOT NULL,
    bank_name VARCHAR(255) NOT NULL,
    primary_product VARCHAR(128),
    product_types TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    product_scores JSONB NOT NULL DEFAULT '{}'::JSONB
        CHECK (jsonb_typeof(product_scores) = 'object'),
    classification_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0
        CHECK (classification_confidence BETWEEN 0.0 AND 1.0),
    classification_status VARCHAR(16) NOT NULL
        CHECK (
            classification_status IN (
                'accepted', 'review', 'required', 'verified'
            )
        ),
    classification_conflict BOOLEAN NOT NULL DEFAULT FALSE,
    page_title TEXT,
    section_heading TEXT,
    source_url TEXT,
    canonical_url TEXT,
    effective_date DATE,
    campaign_start DATE,
    campaign_end DATE,
    content TEXT NOT NULL CHECK (BTRIM(content) <> ''),
    facts JSONB NOT NULL DEFAULT '[]'::JSONB
        CHECK (jsonb_typeof(facts) = 'array'),
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB
        CHECK (jsonb_typeof(metadata) = 'object'),
    embedding_context TEXT NOT NULL,
    embedding_model VARCHAR(150) NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
    qdrant_point_id UUID NOT NULL UNIQUE,
    search_vector TSVECTOR NOT NULL DEFAULT ''::TSVECTOR,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (campaign_end IS NULL OR campaign_start IS NULL
           OR campaign_end >= campaign_start),
    CHECK (
        (
            scope = 'current'
            AND current_source_id IS NOT NULL
            AND historical_source_id IS NULL
            AND document_id = 'current:' || current_source_id::TEXT
        )
        OR
        (
            scope = 'historical'
            AND current_source_id IS NULL
            AND historical_source_id IS NOT NULL
            AND document_id = 'historical:' || historical_source_id::TEXT
        )
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS rag_chunks_current_source_chunk_uidx
    ON rag_chunks (current_source_id, chunk_index)
    WHERE scope = 'current';
CREATE UNIQUE INDEX IF NOT EXISTS rag_chunks_historical_source_chunk_uidx
    ON rag_chunks (historical_source_id, chunk_index)
    WHERE scope = 'historical';
CREATE INDEX IF NOT EXISTS rag_chunks_offer_idx
    ON rag_chunks (offer_id, chunk_index);
CREATE INDEX IF NOT EXISTS rag_chunks_document_idx
    ON rag_chunks (document_id, chunk_index);
CREATE INDEX IF NOT EXISTS rag_chunks_scope_bank_date_idx
    ON rag_chunks (scope, bank_key, effective_date);
CREATE INDEX IF NOT EXISTS rag_chunks_campaign_bounds_idx
    ON rag_chunks (campaign_start, campaign_end)
    WHERE campaign_start IS NOT NULL OR campaign_end IS NOT NULL;
CREATE INDEX IF NOT EXISTS rag_chunks_classification_idx
    ON rag_chunks (classification_status, classification_confidence DESC);
CREATE INDEX IF NOT EXISTS rag_chunks_safe_classification_idx
    ON rag_chunks (classification_status, classification_confidence DESC)
    WHERE NOT classification_conflict;
CREATE INDEX IF NOT EXISTS rag_chunks_product_types_idx
    ON rag_chunks USING GIN (product_types);
CREATE INDEX IF NOT EXISTS rag_chunks_facts_idx
    ON rag_chunks USING GIN (facts JSONB_PATH_OPS);
CREATE INDEX IF NOT EXISTS rag_chunks_search_vector_idx
    ON rag_chunks USING GIN (search_vector);

CREATE OR REPLACE FUNCTION rag_v2_update_search_vector()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $function$
BEGIN
    NEW.search_vector :=
        setweight(
            to_tsvector(
                'simple'::REGCONFIG,
                public.unaccent(COALESCE(NEW.page_title, ''))
            ),
            'A'
        )
        || setweight(
            to_tsvector(
                'simple'::REGCONFIG,
                public.unaccent(
                    COALESCE(NEW.bank_name, '') || ' '
                    || COALESCE(NEW.bank_key, '')
                )
            ),
            'A'
        )
        || setweight(
            to_tsvector(
                'simple'::REGCONFIG,
                public.unaccent(
                    COALESCE(NEW.primary_product, '') || ' '
                    || COALESCE(
                        array_to_string(NEW.product_types, ' '),
                        ''
                    )
                )
            ),
            'B'
        )
        || setweight(
            to_tsvector(
                'simple'::REGCONFIG,
                public.unaccent(COALESCE(NEW.section_heading, ''))
            ),
            'B'
        )
        || setweight(
            to_tsvector(
                'simple'::REGCONFIG,
                public.unaccent(COALESCE(NEW.facts::TEXT, ''))
            ),
            'B'
        )
        || setweight(
            to_tsvector(
                'simple'::REGCONFIG,
                public.unaccent(COALESCE(NEW.content, ''))
            ),
            'C'
        );
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS rag_chunks_search_vector_trigger ON rag_chunks;
CREATE TRIGGER rag_chunks_search_vector_trigger
BEFORE INSERT OR UPDATE OF
    page_title,
    bank_name,
    bank_key,
    primary_product,
    product_types,
    section_heading,
    facts,
    content
ON rag_chunks
FOR EACH ROW
EXECUTE FUNCTION rag_v2_update_search_vector();

-- Rebuild rows left by a partial/manual run before this trigger existed.
UPDATE rag_chunks
SET content = content
WHERE search_vector = ''::TSVECTOR;

CREATE TABLE IF NOT EXISTS rag_sessions (
    id UUID PRIMARY KEY,
    token_hash CHAR(64) NOT NULL UNIQUE
        CHECK (token_hash ~ '^[0-9a-f]{64}$'),
    owner_hash CHAR(64)
        CHECK (owner_hash IS NULL OR owner_hash ~ '^[0-9a-f]{64}$'),
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB
        CHECK (jsonb_typeof(metadata) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    CHECK (expires_at > created_at)
);

CREATE INDEX IF NOT EXISTS rag_sessions_expiry_idx
    ON rag_sessions (expires_at)
    WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS rag_sessions_owner_idx
    ON rag_sessions (owner_hash, expires_at)
    WHERE owner_hash IS NOT NULL AND revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS rag_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES rag_sessions(id) ON DELETE CASCADE,
    turn_id UUID NOT NULL,
    role VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL CHECK (BTRIM(content) <> ''),
    route JSONB CHECK (route IS NULL OR jsonb_typeof(route) = 'object'),
    inherited_context JSONB NOT NULL DEFAULT '{}'::JSONB
        CHECK (jsonb_typeof(inherited_context) = 'object'),
    status VARCHAR(32) CHECK (
        status IS NULL OR status IN (
            'verified',
            'rejected',
            'insufficient_evidence',
            'needs_clarification'
        )
    ),
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB
        CHECK (jsonb_typeof(metadata) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, turn_id, role)
);

CREATE INDEX IF NOT EXISTS rag_messages_session_time_idx
    ON rag_messages (session_id, created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS rag_messages_session_turn_idx
    ON rag_messages (session_id, turn_id);

CREATE TABLE IF NOT EXISTS rag_session_state (
    session_id UUID PRIMARY KEY REFERENCES rag_sessions(id) ON DELETE CASCADE,
    state JSONB NOT NULL DEFAULT '{
        "active_banks": [],
        "active_products": [],
        "active_scope": "current",
        "active_year": null,
        "active_date_from": null,
        "active_date_to": null,
        "active_offer_ids": [],
        "ranked_offers": [],
        "last_intent": null,
        "last_field_types": [],
        "last_source_ids": [],
        "last_document_ids": [],
        "last_standalone_query": null
    }'::JSONB
        CHECK (jsonb_typeof(state) = 'object'),
    version BIGINT NOT NULL DEFAULT 0 CHECK (version >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS rag_session_state_updated_idx
    ON rag_session_state (updated_at);

CREATE TABLE IF NOT EXISTS rag_turn_evidence (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES rag_sessions(id) ON DELETE CASCADE,
    turn_id UUID NOT NULL,
    assistant_message_id BIGINT REFERENCES rag_messages(id)
        ON DELETE SET NULL,
    source_id VARCHAR(32) NOT NULL
        CHECK (source_id ~ '^S[1-9][0-9]*$'),
    chunk_id CHAR(64) REFERENCES rag_chunks(chunk_id)
        ON UPDATE CASCADE ON DELETE SET NULL,
    offer_id CHAR(64) NOT NULL,
    document_id VARCHAR(96) NOT NULL,
    evidence JSONB NOT NULL CHECK (jsonb_typeof(evidence) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, turn_id, source_id)
);

CREATE INDEX IF NOT EXISTS rag_turn_evidence_session_time_idx
    ON rag_turn_evidence (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS rag_turn_evidence_session_turn_idx
    ON rag_turn_evidence (session_id, turn_id);
CREATE INDEX IF NOT EXISTS rag_turn_evidence_offer_idx
    ON rag_turn_evidence (offer_id, document_id);

CREATE OR REPLACE FUNCTION rag_v2_touch_session_state()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $function$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS rag_session_state_touch_trigger ON rag_session_state;
CREATE TRIGGER rag_session_state_touch_trigger
BEFORE UPDATE ON rag_session_state
FOR EACH ROW
EXECUTE FUNCTION rag_v2_touch_session_state();
