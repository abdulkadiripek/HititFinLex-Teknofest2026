-- RAG V2 conversational replies use a distinct non-evidentiary status.
-- Target: PostgreSQL 18. This migration is safe to execute more than once.

DO $migration$
DECLARE
    constraint_name TEXT;
BEGIN
    SELECT constraint_row.conname
    INTO constraint_name
    FROM pg_constraint AS constraint_row
    WHERE constraint_row.conrelid = 'public.rag_messages'::REGCLASS
      AND constraint_row.contype = 'c'
      AND pg_get_constraintdef(constraint_row.oid) ILIKE '%status%'
      AND pg_get_constraintdef(constraint_row.oid) ILIKE '%insufficient_evidence%'
    ORDER BY constraint_row.conname
    LIMIT 1;

    IF constraint_name IS NOT NULL THEN
        EXECUTE format(
            'ALTER TABLE public.rag_messages DROP CONSTRAINT %I',
            constraint_name
        );
    END IF;
END;
$migration$;

ALTER TABLE public.rag_messages
    DROP CONSTRAINT IF EXISTS rag_messages_status_check;

ALTER TABLE public.rag_messages
    ADD CONSTRAINT rag_messages_status_check
    CHECK (
        status IS NULL OR status IN (
            'verified',
            'rejected',
            'insufficient_evidence',
            'needs_clarification',
            'conversational'
        )
    );
