-- Preserve the accepted document revision observed when a review is queued.
-- NULL base_document_exists marks legacy reviews whose baseline is unknown;
-- approval code must reject those rows until they are re-queued explicitly.

ALTER TABLE document_intake_review_queue
    ADD COLUMN base_document_exists BOOLEAN;

ALTER TABLE document_intake_review_queue
    ADD COLUMN base_document_hash CHAR(64);

ALTER TABLE document_intake_review_queue
    ADD CONSTRAINT document_intake_review_base_snapshot_check CHECK (
        (base_document_exists IS NULL AND base_document_hash IS NULL)
        OR (base_document_exists IS FALSE AND base_document_hash IS NULL)
        OR (base_document_exists IS TRUE AND base_document_hash IS NOT NULL)
    );
