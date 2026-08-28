import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const pageSource = await readFile(
  new URL("../app/page.tsx", import.meta.url),
  "utf8",
);
const ragSource = await readFile(
  new URL("../app/rag-v2.ts", import.meta.url),
  "utf8",
);
const frontendSource = `${pageSource}\n${ragSource}`;

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
  assert.match(ragSource, /parsed\.protocol !== "http:" && parsed\.protocol !== "https:"/);
  assert.match(ragSource, /parsed\.username \|\| parsed\.password/);
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
  assert.match(pageSource, /messageSourceAnchor\(messageId, sourceId\)/);
  assert.match(pageSource, /id={messageSourceAnchor\(message\.id, sourceId\)}/);
  assert.match(ragSource, /`message-\${messageId}-source-\${normalized}`/);
});

test("labels source provenance neutrally and links headline metrics to their exact document", () => {
  assert.match(pageSource, /Otomatik sınıflandırılmış kaynak/);
  assert.match(pageSource, /Otomatik işlenmiş kaynak/);
  assert.match(pageSource, /Kaynak metni mevcut/);
  assert.doesNotMatch(pageSource, /insan doğrulaması yok/i);
  assert.match(pageSource, /className="metric-source-button"/);
  assert.match(
    pageSource,
    /document_id: primary\?\.document_id \?\? item\.document_id/,
  );
});

test("wires the RAG V2 session protocol without browser-side secrets", () => {
  assert.match(frontendSource, /\/rag\/v2\/chat/);
  assert.match(pageSource, /\/rag\/v2\/sessions/);
  assert.match(pageSource, /\/clear/);
  assert.match(frontendSource, /X-RAG-Client-Id/);
  assert.match(pageSource, /window\.localStorage/);
  assert.match(pageSource, /window\.sessionStorage/);
  assert.match(pageSource, /initializePersistentRagStorage/);
  assert.match(pageSource, /\/rag\/v2\/session\/messages/);
  assert.match(pageSource, /parseRagSessionMessagesResponse/);
  assert.match(pageSource, /setConversation\(restored\)/);
  assert.match(pageSource, /messageId\.current = restored\.length \+ 1/);
  assert.match(pageSource, /requestRagChatWithRetry/);
  assert.match(ragSource, /use_reranker: false/);
  assert.match(pageSource, /"X-RAG-Session-Id": sessionId/);
  assert.match(pageSource, /ragApiUrl\("\/rag\/v2\/session\/clear"\)/);
  assert.doesNotMatch(pageSource, /encodeURIComponent\(sessionId\)/);
  assert.doesNotMatch(pageSource, /\/session\/messages\?\S*session/i);
  assert.match(pageSource, /top_k: 12/);
  assert.doesNotMatch(
    frontendSource,
    /EVREN_API_KEY|QDRANT_API_KEY|sk-evren-|qdr-team\d+-|Arayüz Parola/,
  );
});

test("aborts and invalidates stale chat work before session mutations", () => {
  assert.match(pageSource, /const chatAbort = useRef<AbortController \| null>/);
  assert.match(pageSource, /const chatRequestId = useRef\(0\)/);
  assert.match(pageSource, /const chatBusyRef = useRef\(false\)/);
  assert.match(pageSource, /const sessionActionBusyRef = useRef\(false\)/);
  assert.match(pageSource, /const SESSION_ACTION_TIMEOUT_MS = 30_000/);
  assert.match(pageSource, /chatRequestId\.current \+= 1/);
  assert.match(pageSource, /chatAbort\.current\?\.abort\(\)/);
  assert.match(pageSource, /requestId !== chatRequestId\.current \|\| controller\.signal\.aborted/);
  assert.match(pageSource, /async function startNewChat\(\)[\s\S]*?invalidateChatRequest\(\)/);
  assert.match(pageSource, /async function clearConversationContext\(\)[\s\S]*?invalidateChatRequest\(\)/);
  assert.match(pageSource, /result\.retriedWithoutExpiredSession \? \[conversationItem\] : \[\.\.\.current, conversationItem\]/);

  const clearStart = pageSource.indexOf("async function clearConversationContext()");
  const clearEnd = pageSource.indexOf("async function submitChat", clearStart);
  const clearBody = pageSource.slice(clearStart, clearEnd);
  assert.ok(clearStart >= 0 && clearEnd > clearStart);
  assert.ok(clearBody.indexOf("await clearRagSession") < clearBody.indexOf("setConversation([])"));
});

test("exposes verified, fail-closed, and clarification states", () => {
  assert.match(pageSource, /statusLabel\(displayStatus\)/);
  assert.match(pageSource, /displayAnswer\(message\)/);
  assert.match(pageSource, /responseStatusForDisplay\(message\)/);
  assert.match(pageSource, /buildInheritedContextChips/);
  assert.match(ragSource, /response\.status === "needs_clarification"/);
  assert.match(ragSource, /response\.status !== "verified"/);
  assert.match(ragSource, /SAFE_FALLBACK_ANSWER/);
});
