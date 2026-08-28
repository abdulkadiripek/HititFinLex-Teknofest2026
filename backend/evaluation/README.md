# RAG V2 offline evaluation

This directory contains a manually authored multi-turn routing set and a
provider-neutral comparison utility. The dataset label is exactly
`silver_unverified`. It is not a gold set and it does not contain verified
document relevance judgments or live benchmark scores.

## Assets

- `multiturn_scenarios.silver_unverified.json`: 32 scenarios and 66 turns.
- `retrieval_cases.silver_unverified.json`: 32 fixed current-corpus queries.
- `rag_v2_metrics.py`: schema validation and deterministic metric formulas.
- `evaluate_rag_v2.py`: standalone JSON report command.
- `router_compare.py`: live deterministic QueryRouter/no-memory comparison.
- `retrieval_compare.py`: same-set live legacy/V2 retrieval runner.
- `secret_hygiene.py`: redacted source, runtime-log, and Git-history scan.

The scenarios cover bank/product/year/date inheritance, deterministic ordinal
offer references, topic and scope changes, ambiguous references, no-data
questions, untrusted prior assistant answers, fresh retrieval requirements,
and cross-session isolation probes. `eval:*` offer ids and `eval_bank_*` names
are harness references, not claims about production records.

## Run without provider outputs

```powershell
python -m evaluation.evaluate_rag_v2
```

This intentionally reports metrics as `unavailable`. It never invents a
score when provider outputs or relevance labels are missing.

## Deterministic multi-turn router comparison

Run all 32 scenarios and 66 turns directly against the checked-in
`rag_v2.routing.QueryRouter` implementation:

```powershell
python -m evaluation.router_compare `
  --output evaluation\router-comparison.local.json
```

The V2 adapter keeps a separate `SessionState` for each scenario
`session_key`. A `context_after_turn` fixture is injected only into that
session after the corresponding route is evaluated. These fixtures model
verified retrieval state; simulated assistant answers are deliberately
ignored. The comparison baseline passes the raw query through with no
server-side memory.

The report includes both normalized-exact standalone-query accuracy and a
separate context-coverage accuracy for non-clarification turns. Context
coverage requires every labeled bank, product, field, date boundary, and
scope marker to occur in the generated standalone query; it does not replace
or hide the stricter exact score. The report also includes inherited
bank/product/date/scope accuracy, complete inheritance accuracy, topic-clear
accuracy, and clarification accuracy. Empty inheritance labels are excluded
from inheritance denominators. Topic clearing is inferred from actual state
transitions rather than copied from expected labels. Retrieval, citation,
numeric, rejection, and answer-level metrics are marked `unavailable`.
Session isolation is measured on the explicit cross-session probes by
requiring the actual bank, product, year, and offer state to match the fresh
session label; the stateless baseline also passes when it carries no state.
Metrics outside routing are marked `unavailable`
because this command does not execute those layers. Every result remains
`silver_unverified`; the report explicitly disallows a production-quality
claim.

## Paired comparison input

Pass one JSON object with `dataset_label` and `records`. Every record has one
shared `labels` object and optional `legacy` and `v2` outputs:

```json
{
  "schema_version": "1.0",
  "dataset_label": "silver_unverified",
  "records": [
    {
      "record_id": "mt_001_bank_product_inheritance/t1",
      "labels": {
        "relevant_ids": ["document-id-from-human-review"],
        "relevance_grades": {"document-id-from-human-review": 2},
        "citation": {
          "required_ids": ["document-id-from-human-review"],
          "allowed_ids": ["document-id-from-human-review"]
        },
        "numbers": {
          "required": ["36 ay"],
          "allowed": ["36 ay"]
        }
      },
      "legacy": {
        "standalone_query": "...",
        "inherited_context": {},
        "needs_clarification": false,
        "retrieved_ids": ["document-id-from-human-review"],
        "cited_ids": ["document-id-from-human-review"],
        "answer_numbers": ["36 ay"],
        "status": "verified"
      },
      "v2": {
        "standalone_query": "...",
        "inherited_context": {},
        "needs_clarification": false,
        "retrieved_ids": ["document-id-from-human-review"],
        "cited_ids": ["document-id-from-human-review"],
        "answer_numbers": ["36 ay"],
        "status": "verified"
      }
    }
  ]
}
```

The identifiers and number above only illustrate the input shape. Replace
them with human-reviewed labels; do not treat the example as a fact.

Run and save the report:

```powershell
python -m evaluation.evaluate_rag_v2 `
  --comparison path\to\paired-results.json `
  --output path\to\report.json
```

Supplemental labels may only fill fields absent from the scenario set. A
conflicting route label is rejected. Unknown or duplicate record ids are also
rejected.

## Fairness and metric definitions

For each metric, every labeled record must contain both a legacy result and a
V2 result. If either result is missing, the metric and its delta are
`unavailable`; the utility does not compare selective subsets.

- Standalone query accuracy uses normalized exact match.
- Bank, product, date, and scope inheritance metrics use exact set/value match
  for inherited fields.
- Topic-clear accuracy uses the exact cleared-field set on every labeled turn,
  including empty sets, so false-positive context clearing is penalized.
- Clarification accuracy is boolean classification accuracy.
- Recall@K is relevant retrieved ids divided by all relevant ids.
- MRR@10 uses the first relevant result.
- nDCG@10 uses graded labels when present and binary relevance otherwise.
- Citation accuracy requires all required ids and rejects ids outside the
  allowed set.
- Numeric accuracy requires all required normalized values and rejects values
  outside the allowed set.
- Unsupported rejection accepts `rejected` or `insufficient_evidence`.
- Isolation accuracy compares an explicit `isolation_passed` probe result.

The JSON report always states that no improvement claim is made. A measured
delta can only be interpreted after the silver labels have been reviewed and
the same complete provider outputs have been supplied.

## Reproducible live retrieval comparison

The fixed retrieval set uses the stable `kayit_id`, bank key, page title, and
canonical source URL from `data/ara/dokumanlar.jsonl`. Each case is labeled
`silver_unverified`; exact-title/source matching is useful for repeatability
but is not a substitute for human gold relevance review. The intended source
is labeled for each exact-page query; unjudged documents are not asserted to
be irrelevant.

Validate all 32 labels against the checked-in corpus without calling a model,
database, EVREN, or Qdrant:

```powershell
python -m evaluation.retrieval_compare --validate-only
```

Run both live retrievers on the same ordered 32 cases:

```powershell
python -m evaluation.retrieval_compare `
  --output evaluation\retrieval-comparison.local.json
```

The generated report contains Recall@1/3/5/10, MRR@10, nDCG@10, the exact
record ids returned for every case, dataset and corpus SHA-256 values, and
paired V2-minus-legacy deltas. `top_k` cannot be below 10.

The legacy adapter calls the existing `hybrid_search.py` BGE-M3/PostgreSQL
retriever. The V2 adapter calls `rag_v2.retrieval.HybridRetriever` with the
case's fixed structured route, EVREN embeddings, Qdrant dense retrieval, and
PostgreSQL lexical retrieval. It rejects a lexical-only V2 fallback: every
case must have dense candidates before the provider is considered available.

Prerequisites for a measured comparison:

- the current PostgreSQL `documents` and `document_chunks` index;
- the migrated and populated `rag_chunks` table;
- the configured and populated RAG V2 Qdrant collection;
- EVREN embedding access;
- the legacy BGE-M3 model available locally or downloadable.

If the corpus, a label, a table, model, or remote service is unavailable, the
paired metrics are emitted as `unavailable`. Reports contain fixed reason
codes only; raw connection errors, API keys, passwords, and provider response
bodies are never included.

## Secret hygiene

Run the repository scan before packaging or committing:

```powershell
python -m evaluation.secret_hygiene
```

The scanner covers tracked and non-ignored source assets, runtime `*.log`
files, and every reachable Git snapshot. It detects EVREN and Qdrant key
shapes, credential-bearing PostgreSQL URLs, bearer literals, quoted secret
assignments, and unquoted uppercase environment assignments. Findings contain
only a detector name, path, and optional commit id; matched values and source
lines are never returned. The ignored root `.env` is the only exempt dotenv
file. A tracked nested `.env` or `.env.*` file is scanned.
