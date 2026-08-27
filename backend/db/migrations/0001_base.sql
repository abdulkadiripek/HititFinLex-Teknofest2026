-- HititFinLex PostgreSQL baseline schema
-- Target: PostgreSQL 18 + pgvector 0.8.6

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS banks (
    id BIGSERIAL PRIMARY KEY,
    bank_key VARCHAR(100) NOT NULL UNIQUE,
    bank_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    record_key VARCHAR(200) NOT NULL UNIQUE,
    bank_id BIGINT NOT NULL REFERENCES banks(id),
    source_url TEXT NOT NULL UNIQUE,
    page_title TEXT,
    raw_text TEXT NOT NULL,
    summary_text TEXT,
    campaign_type_code VARCHAR(100),
    campaign_type VARCHAR(255),
    confidence DOUBLE PRECISION CHECK (
        confidence IS NULL OR confidence BETWEEN 0 AND 1
    ),
    label_source VARCHAR(100),
    rationale TEXT,
    verified BOOLEAN NOT NULL DEFAULT FALSE,
    auto_accepted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_bank
    ON documents (bank_id);
CREATE INDEX IF NOT EXISTS idx_documents_campaign_type
    ON documents (campaign_type_code);

CREATE TABLE IF NOT EXISTS passages (
    id BIGSERIAL PRIMARY KEY,
    passage_key VARCHAR(220) NOT NULL UNIQUE,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    campaign_type_code VARCHAR(100),
    content TEXT NOT NULL,
    document_offset INTEGER CHECK (
        document_offset IS NULL OR document_offset >= 0
    ),
    verified BOOLEAN NOT NULL DEFAULT FALSE,
    auto_accepted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_passages_document
    ON passages (document_id);

CREATE TABLE IF NOT EXISTS entities (
    id BIGSERIAL PRIMARY KEY,
    passage_id BIGINT NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    span_index INTEGER NOT NULL,
    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset INTEGER NOT NULL,
    entity_label VARCHAR(100) NOT NULL,
    entity_text TEXT NOT NULL,
    normalized_value JSONB,
    confidence DOUBLE PRECISION CHECK (
        confidence IS NULL OR confidence BETWEEN 0 AND 1
    ),
    extraction_source VARCHAR(100),
    verified BOOLEAN NOT NULL DEFAULT FALSE,
    review_note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (end_offset > start_offset),
    UNIQUE (passage_id, span_index)
);

CREATE INDEX IF NOT EXISTS idx_entities_passage
    ON entities (passage_id);
CREATE INDEX IF NOT EXISTS idx_entities_label
    ON entities (entity_label);

CREATE TABLE IF NOT EXISTS classification_samples (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    dataset_version VARCHAR(50) NOT NULL DEFAULT 'v1',
    data_split VARCHAR(10) NOT NULL CHECK (data_split IN ('train', 'val', 'test')),
    label_name VARCHAR(255) NOT NULL,
    label_code VARCHAR(100) NOT NULL,
    verified BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, dataset_version)
);

CREATE INDEX IF NOT EXISTS idx_classification_split
    ON classification_samples (data_split);
CREATE INDEX IF NOT EXISTS idx_classification_label
    ON classification_samples (label_code);

CREATE TABLE IF NOT EXISTS extraction_samples (
    id BIGSERIAL PRIMARY KEY,
    passage_id BIGINT NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    dataset_version VARCHAR(50) NOT NULL DEFAULT 'v1',
    data_split VARCHAR(10) NOT NULL CHECK (data_split IN ('train', 'val', 'test')),
    verified BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (passage_id, dataset_version)
);

CREATE INDEX IF NOT EXISTS idx_extraction_split
    ON extraction_samples (data_split);

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    token_count INTEGER CHECK (token_count IS NULL OR token_count > 0),
    content_hash VARCHAR(64),
    embedding_model VARCHAR(150) DEFAULT 'BAAI/bge-m3',
    embedding vector(1024),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('simple', content)
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document
    ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_text_search
    ON document_chunks USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS comparison_facts (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    fact_type VARCHAR(64) NOT NULL,
    fact_text TEXT NOT NULL,
    normalized_value JSONB,
    evidence_text TEXT NOT NULL,
    extraction_method VARCHAR(64) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    source_chunk INTEGER NOT NULL,
    fact_key CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, fact_key)
);

CREATE INDEX IF NOT EXISTS comparison_facts_document_type_idx
    ON comparison_facts (document_id, fact_type);

CREATE TABLE IF NOT EXISTS comparison_fact_review_queue (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    fact_type VARCHAR(64) NOT NULL,
    fact_text TEXT NOT NULL,
    normalized_value JSONB,
    evidence_text TEXT NOT NULL,
    extraction_method VARCHAR(64) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    source_chunk INTEGER NOT NULL,
    fact_key CHAR(64) NOT NULL,
    review_reason VARCHAR(128) NOT NULL,
    review_status VARCHAR(16) NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, fact_key)
);

CREATE INDEX IF NOT EXISTS comparison_fact_review_status_idx
    ON comparison_fact_review_queue (review_status, confidence DESC);

CREATE TABLE IF NOT EXISTS ner_document_state (
    document_id BIGINT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    content_hash CHAR(64) NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    pipeline_version VARCHAR(64) NOT NULL,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    review_count INTEGER NOT NULL DEFAULT 0,
    rejected_count INTEGER NOT NULL DEFAULT 0,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_intake_review_queue (
    id BIGSERIAL PRIMARY KEY,
    record_key VARCHAR(255) NOT NULL,
    bank_key VARCHAR(128) NOT NULL,
    bank_name VARCHAR(255) NOT NULL,
    source_url TEXT NOT NULL,
    page_title TEXT,
    raw_text TEXT NOT NULL,
    content_hash CHAR(64) NOT NULL,
    classification JSONB NOT NULL,
    review_reason VARCHAR(128) NOT NULL,
    review_status VARCHAR(16) NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (record_key, content_hash)
);

CREATE INDEX IF NOT EXISTS document_intake_review_status_idx
    ON document_intake_review_queue (review_status, created_at);

CREATE TABLE IF NOT EXISTS document_intake_state (
    document_id BIGINT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    bank_id BIGINT NOT NULL REFERENCES banks(id) ON DELETE CASCADE,
    record_key VARCHAR(255) NOT NULL,
    content_hash CHAR(64) NOT NULL,
    pipeline_version VARCHAR(64) NOT NULL,
    classification JSONB NOT NULL,
    accepted_fact_count INTEGER NOT NULL DEFAULT 0,
    review_fact_count INTEGER NOT NULL DEFAULT 0,
    rejected_fact_count INTEGER NOT NULL DEFAULT 0,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (bank_id, content_hash)
);

CREATE TABLE IF NOT EXISTS historical_documents (
    id BIGSERIAL PRIMARY KEY,
    archive_key VARCHAR(255) NOT NULL UNIQUE,
    bank_id BIGINT NOT NULL REFERENCES banks(id) ON DELETE RESTRICT,
    source_url TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    canonical_group_key CHAR(64) NOT NULL,
    archive_url TEXT,
    page_title TEXT,
    raw_text TEXT NOT NULL,
    content_hash CHAR(64) NOT NULL UNIQUE,
    snapshot_date DATE,
    collected_at TIMESTAMPTZ,
    source_category VARCHAR(128),
    is_campaign_hint BOOLEAN NOT NULL DEFAULT FALSE,
    campaign_label VARCHAR(16),
    campaign_confidence DOUBLE PRECISION,
    product_type_code VARCHAR(64),
    product_type VARCHAR(128),
    classification_confidence DOUBLE PRECISION,
    classification_decision VARCHAR(16) NOT NULL
        CHECK (classification_decision IN ('ACCEPTED', 'REVIEW', 'FAILED')),
    classification_basis VARCHAR(128),
    classification_payload JSONB,
    quality_status VARCHAR(16) NOT NULL
        CHECK (quality_status IN ('accepted', 'review', 'failed')),
    searchable BOOLEAN NOT NULL DEFAULT FALSE,
    verified BOOLEAN NOT NULL DEFAULT FALSE,
    verification_source VARCHAR(64),
    pipeline_version VARCHAR(64) NOT NULL,
    source_dataset VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS historical_documents_url_date_idx
    ON historical_documents (canonical_group_key, snapshot_date);
CREATE INDEX IF NOT EXISTS historical_documents_filter_idx
    ON historical_documents (searchable, product_type_code, bank_id, snapshot_date);

CREATE TABLE IF NOT EXISTS historical_facts (
    id BIGSERIAL PRIMARY KEY,
    historical_document_id BIGINT NOT NULL
        REFERENCES historical_documents(id) ON DELETE CASCADE,
    fact_type VARCHAR(64) NOT NULL,
    fact_text TEXT NOT NULL,
    normalized_value JSONB,
    evidence_text TEXT NOT NULL,
    extraction_method VARCHAR(64) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    source_chunk INTEGER NOT NULL DEFAULT 0,
    decision VARCHAR(16) NOT NULL
        CHECK (decision IN ('accepted', 'review', 'rejected')),
    decision_reason VARCHAR(128) NOT NULL,
    review_status VARCHAR(16) NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected', 'not_required')),
    fact_key CHAR(64) NOT NULL,
    pipeline_version VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (historical_document_id, fact_key)
);

CREATE INDEX IF NOT EXISTS historical_facts_document_type_idx
    ON historical_facts (historical_document_id, fact_type, decision);
CREATE INDEX IF NOT EXISTS historical_facts_review_idx
    ON historical_facts (review_status, confidence DESC)
    WHERE decision = 'review';

CREATE TABLE IF NOT EXISTS historical_document_chunks (
    id BIGSERIAL PRIMARY KEY,
    historical_document_id BIGINT NOT NULL
        REFERENCES historical_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER,
    content_hash CHAR(64) NOT NULL,
    embedding_model VARCHAR(128),
    embedding vector(1024),
    search_vector TSVECTOR,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (historical_document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS historical_chunks_search_idx
    ON historical_document_chunks USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS historical_chunks_embedding_idx
    ON historical_document_chunks USING hnsw (embedding vector_cosine_ops);
