import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const pageSource = await readFile(
  new URL("../app/page.tsx", import.meta.url),
  "utf8",
);

test("does not fall back to fixed financial demo records", () => {
  assert.doesNotMatch(
    pageSource,
    /demoDashboard|demoCatalog|demoComparison|dataset_snapshot/,
  );
  assert.match(pageSource, /sabit finansal örnek veri gösterilmiyor/);
});

test("maps the card product code per API scope", () => {
  assert.match(
    pageSource,
    /normalized === "KART_KAMPANYASI" \? "KART" : normalized/,
  );
  assert.match(
    pageSource,
    /scope === "history" \? "KART_KAMPANYASI" : "KART"/,
  );
});

test("guards scoped requests and forwards historical quality filters", () => {
  assert.match(pageSource, /catalogRequestId/);
  assert.match(pageSource, /catalogRequestId.current === bootCatalogRequestId/);
  assert.match(pageSource, /compareRequestId/);
  assert.match(pageSource, /catalogAbort\.current\?\.abort\(\)/);
  assert.match(pageSource, /compareAbort\.current\?\.abort\(\)/);
  assert.match(
    pageSource,
    /product_types: effectiveProduct \? \[apiProductCode\(effectiveProduct, "history"\)\] : \[\],[\s\S]*has_facts: hasFacts,[\s\S]*min_confidence: Number\(effectiveConfidence\)/,
  );
});

test("uses the active LLM health and allows only HTTP(S) source links", () => {
  assert.match(pageSource, /healthData\?\.llm_model_ready === false/);
  assert.match(pageSource, /function safeExternalUrl/);
  assert.match(
    pageSource,
    /parsed\.protocol === "http:" \|\| parsed\.protocol === "https:"/,
  );
  assert.match(pageSource, /function sanitizeExternalUrls/);
  assert.match(
    pageSource,
    /key === "source_url" \|\| key === "archive_url"/,
  );
  assert.match(
    pageSource,
    /health\?\.active_model \?\? health\?\.ollama_model/,
  );
});

test("uses message-local citation ids", () => {
  assert.match(
    pageSource,
    /href={`#message-\${messageId}-source-\${cite\[1\]}`}/,
  );
  assert.match(
    pageSource,
    /id={`message-\${message\.id}-source-\${source\.source_id}`}/,
  );
});

test("labels model sources and links headline metrics to their exact document", () => {
  assert.match(pageSource, /Model kaynağı · insan doğrulaması yok/);
  assert.match(pageSource, /Model · insan doğrulaması yok/);
  assert.match(pageSource, /className="metric-source-button"/);
  assert.match(
    pageSource,
    /document_id: primary\?\.document_id \?\? item\.document_id/,
  );
});
