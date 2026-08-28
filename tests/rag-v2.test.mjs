import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  RAG_CLIENT_STORAGE_KEY,
  RAG_SESSION_STORAGE_KEY,
  SAFE_FALLBACK_ANSWER,
  answerTokens,
  buildInheritedContextChips,
  citationIdFromToken,
  clientHeaders,
  displayAnswer,
  evidenceSourceId,
  getOrCreateClientId,
  initializePersistentRagStorage,
  knownCitationIds,
  loadSessionId,
  messageSourceAnchor,
  parseRagSessionMessagesResponse,
  parseRagV2Response,
  requestRagChatWithRetry,
  responseStatusForDisplay,
  safeHttpUrl,
  saveSessionId,
} from "../app/rag-v2.ts";

const pageSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

class MemoryStorage {
  values = new Map();

  getItem(key) {
    return this.values.get(key) ?? null;
  }

  setItem(key, value) {
    this.values.set(key, String(value));
  }

  removeItem(key) {
    this.values.delete(key);
  }
}

const SESSION_ID = "session-opaque-1234567890";
const CLIENT_UUID = "11111111-2222-4333-8444-555555555555";

function response(overrides = {}) {
  return {
    session_id: SESSION_ID,
    query: "Peki vadesi?",
    standalone_query: "Ziraat Katilim konut finansmani vadesi",
    answer: "Vade 120 aydir [S1].",
    status: "verified",
    inherited_context: { banks: ["Ziraat Katilim"], inherited_fields: ["banks"] },
    route: { banks: ["Ziraat Katilim"], product_types: ["KONUT_FINANSMANI"], scope: "current" },
    evidence: [{ source_id: "S1", evidence_text: "Azami vade 120 aydir." }],
    issues: [],
    diagnostics: {},
    ...overrides,
  };
}

test("keeps opaque session and random client identity in persistent storage", () => {
  const storage = new MemoryStorage();
  const clientId = getOrCreateClientId(storage, () => CLIENT_UUID);
  assert.equal(clientId, `rag-web-${CLIENT_UUID}`);
  assert.equal(storage.getItem(RAG_CLIENT_STORAGE_KEY), clientId);
  assert.equal(getOrCreateClientId(storage, () => "should-not-run"), clientId);

  saveSessionId(storage, SESSION_ID);
  assert.equal(storage.getItem(RAG_SESSION_STORAGE_KEY), SESSION_ID);
  assert.equal(loadSessionId(storage), SESSION_ID);

  storage.setItem(RAG_SESSION_STORAGE_KEY, "bad\r\nheader");
  assert.equal(loadSessionId(storage), null);
  assert.equal(storage.getItem(RAG_SESSION_STORAGE_KEY), null);
});

test("migrates a valid legacy session pair once and removes transient copies", () => {
  const persistent = new MemoryStorage();
  const legacy = new MemoryStorage();
  const clientId = `rag-web-${CLIENT_UUID}`;
  legacy.setItem(RAG_CLIENT_STORAGE_KEY, clientId);
  legacy.setItem(RAG_SESSION_STORAGE_KEY, SESSION_ID);

  const restored = initializePersistentRagStorage(persistent, legacy, () => "should-not-run");

  assert.deepEqual(restored, { clientId, sessionId: SESSION_ID });
  assert.equal(persistent.getItem(RAG_CLIENT_STORAGE_KEY), clientId);
  assert.equal(persistent.getItem(RAG_SESSION_STORAGE_KEY), SESSION_ID);
  assert.equal(legacy.getItem(RAG_CLIENT_STORAGE_KEY), null);
  assert.equal(legacy.getItem(RAG_SESSION_STORAGE_KEY), null);
});

test("prefers a complete persistent pair and never adopts an ownerless session", () => {
  const persistent = new MemoryStorage();
  const legacy = new MemoryStorage();
  const persistentClient = `rag-web-${CLIENT_UUID}`;
  const legacyClient = "rag-web-99999999-2222-4333-8444-555555555555";
  persistent.setItem(RAG_CLIENT_STORAGE_KEY, persistentClient);
  persistent.setItem(RAG_SESSION_STORAGE_KEY, SESSION_ID);
  legacy.setItem(RAG_CLIENT_STORAGE_KEY, legacyClient);
  legacy.setItem(RAG_SESSION_STORAGE_KEY, "legacy-session-opaque-1234567890");

  assert.deepEqual(initializePersistentRagStorage(persistent, legacy), {
    clientId: persistentClient,
    sessionId: SESSION_ID,
  });

  const ownerlessPersistent = new MemoryStorage();
  ownerlessPersistent.setItem(RAG_SESSION_STORAGE_KEY, SESSION_ID);
  const fresh = initializePersistentRagStorage(ownerlessPersistent, new MemoryStorage(), () => CLIENT_UUID);
  assert.deepEqual(fresh, { clientId: persistentClient, sessionId: null });
  assert.equal(ownerlessPersistent.getItem(RAG_SESSION_STORAGE_KEY), null);
});

test("rejects header injection and non-http source URLs", () => {
  assert.throws(() => clientHeaders("bad\r\nvalue"), /Invalid client id/);
  assert.equal(safeHttpUrl("javascript:alert(1)"), undefined);
  assert.equal(safeHttpUrl("data:text/html,unsafe"), undefined);
  assert.equal(safeHttpUrl("https://user:pass@example.com/source"), undefined);
  assert.equal(safeHttpUrl("https://example.com/source"), "https://example.com/source");
  assert.equal(safeHttpUrl("http://example.com/source"), "http://example.com/source");
});

test("normalizes S citations and creates message-local anchors", () => {
  const evidence = [{ source_id: 1 }, { source_id: "S2" }];
  assert.equal(evidenceSourceId(evidence[0], 0), "S1");
  assert.deepEqual([...knownCitationIds(evidence)], ["S1", "S2"]);
  assert.equal(citationIdFromToken("[s2]"), "S2");
  assert.equal(citationIdFromToken("[S99]"), "S99");
  assert.equal(citationIdFromToken("[1]"), null);
  assert.notEqual(messageSourceAnchor(1, "S1"), messageSourceAnchor(2, "S1"));
  assert.deepEqual(answerTokens("Bilgi [S1], bilinmeyen [S9]."), ["Bilgi ", "[S1]", ", bilinmeyen ", "[S9]", "."]);
});

test("renders only verified evidence-backed answers and fails closed otherwise", () => {
  assert.equal(displayAnswer(response()), "Vade 120 aydir [S1].");
  assert.equal(responseStatusForDisplay(response()), "verified");
  assert.equal(displayAnswer(response({ evidence: [] })), SAFE_FALLBACK_ANSWER);
  assert.equal(displayAnswer(response({ answer: "Vade 120 aydir." })), SAFE_FALLBACK_ANSWER);
  assert.equal(displayAnswer(response({ answer: "Vade 120 aydir [S99]." })), SAFE_FALLBACK_ANSWER);
  assert.equal(responseStatusForDisplay(response({ answer: "Vade 120 aydir [S99]." })), "rejected");
  assert.equal(displayAnswer(response({ answer: "Vade 120 aydir [S1] [S0]." })), SAFE_FALLBACK_ANSWER);
  assert.equal(displayAnswer(response({ status: "rejected", answer: "Unsupported 999" })), SAFE_FALLBACK_ANSWER);
  assert.equal(displayAnswer(response({ status: "insufficient_evidence", answer: "Unsupported 999" })), SAFE_FALLBACK_ANSWER);
  assert.equal(
    displayAnswer(response({
      status: "needs_clarification",
      answer: "Unsupported 999",
      route: { clarification_question: "Hangi bankayi kastediyorsunuz?" },
      evidence: [],
    })),
    "Hangi bankayi kastediyorsunuz?",
  );
  assert.equal(
    displayAnswer(response({
      status: "needs_clarification",
      answer: "Hangi urunu kastediyorsunuz?",
      route: { needs_clarification: true, clarification_question: null },
      evidence: [],
    })),
    "Hangi urunu kastediyorsunuz?",
  );
});

test("accepts conversational answers only on the evidence-free chat route", () => {
  const conversational = response({
    status: "conversational",
    answer: "Merhaba, kaldığımız yerden devam edebiliriz.",
    route: { intent: "chat", needs_clarification: false },
    evidence: [],
  });

  assert.ok(parseRagV2Response(conversational));
  assert.equal(displayAnswer(conversational), conversational.answer);
  assert.equal(responseStatusForDisplay(conversational), "conversational");
  assert.equal(displayAnswer({ ...conversational, answer: "Eski kaynak [S1]." }), SAFE_FALLBACK_ANSWER);
  assert.equal(responseStatusForDisplay({ ...conversational, answer: "Eski kaynak [S1]." }), "rejected");
  assert.equal(parseRagV2Response({ ...conversational, route: { intent: "lookup", needs_clarification: false } }), null);
  assert.equal(parseRagV2Response({ ...conversational, evidence: [{ source_id: "S1", evidence_text: "Finansal kanit." }] }), null);
});

test("shows only inherited bank, product, year, and scope context", () => {
  const chips = buildInheritedContextChips({
    banks: ["Ziraat Katilim"],
    product_types: ["KONUT_FINANSMANI"],
    year: 2025,
    scope: "historical",
    inherited_fields: ["banks", "product_types", "year", "scope"],
  });
  assert.deepEqual(chips.map((chip) => chip.kind), ["bank", "product", "year", "scope"]);
  assert.equal(chips.at(-1).value, "Tarihsel");
});

test("rejects malformed RAG responses before rendering", () => {
  assert.ok(parseRagV2Response(response()));
  assert.equal(parseRagV2Response(response({ session_id: "short" })), null);
  assert.equal(parseRagV2Response(response({ status: "unknown" })), null);
  assert.equal(parseRagV2Response(response({ evidence: [{ content: "missing id" }] })), null);
  assert.equal(parseRagV2Response(response({ evidence: [{ source_id: "bad-id" }] })), null);
  assert.equal(parseRagV2Response(response({ evidence: [{ source_id: "S1" }, { source_id: "s1" }] })), null);
  assert.equal(parseRagV2Response(response({ issues: ["safe", { detail: "invalid" }] })), null);
  assert.equal(parseRagV2Response(response({
    status: "needs_clarification",
    route: { clarification_question: "Hangi banka?", needs_clarification: false },
    evidence: [],
  })), null);
});

test("parses only owner-scoped session transcript response shapes", () => {
  const parsed = parseRagSessionMessagesResponse({
    session_id: SESSION_ID,
    expires_at: "2026-08-29T00:00:00Z",
    messages: [response(), response({ query: "Peki tutari?" })],
  });
  assert.equal(parsed?.messages.length, 2);
  assert.equal(parseRagSessionMessagesResponse({ session_id: "short", messages: [] }), null);
  assert.equal(parseRagSessionMessagesResponse({ session_id: SESSION_ID, messages: [response({ status: "bad" })] }), null);
  assert.equal(parseRagSessionMessagesResponse({ session_id: SESSION_ID, messages: "not-an-array" }), null);
});

test("retries an expired session once with null and preserves client ownership", async () => {
  const calls = [];
  const fetcher = async (_input, init) => {
    calls.push(init);
    if (calls.length === 1) return new Response("expired", { status: 410 });
    return Response.json(response());
  };
  const result = await requestRagChatWithRetry({
    fetcher,
    apiBaseUrl: "https://api.example.test/v1/",
    clientId: `rag-web-${CLIENT_UUID}`,
    sessionId: SESSION_ID,
    request: { query: "Peki vadesi?", top_k: 12, use_reranker: false },
  });

  assert.equal(result.retriedWithoutExpiredSession, true);
  assert.equal(calls.length, 2);
  assert.equal(JSON.parse(calls[0].body).session_id, SESSION_ID);
  assert.equal(JSON.parse(calls[1].body).session_id, null);
  assert.equal(JSON.parse(calls[1].body).use_reranker, false);
  assert.equal(calls[0].headers["X-RAG-Client-Id"], calls[1].headers["X-RAG-Client-Id"]);
});

test("does not retry authorization errors or expose response bodies", async () => {
  let callCount = 0;
  const secretMarker = "server-secret-marker";
  await assert.rejects(
    requestRagChatWithRetry({
      fetcher: async () => {
        callCount += 1;
        return new Response(secretMarker, { status: 403 });
      },
      apiBaseUrl: "https://api.example.test",
      clientId: `rag-web-${CLIENT_UUID}`,
      sessionId: SESSION_ID,
      request: { query: "test", top_k: 12, use_reranker: false },
    }),
    (error) => {
      assert.doesNotMatch(error.message, new RegExp(secretMarker));
      return true;
    },
  );
  assert.equal(callCount, 1);
});

test("resets assistant filters for new and cleared conversations", () => {
  assert.match(pageSource, /const DEFAULT_ASSISTANT_SCOPE: CompareScope = "live"/);
  assert.match(pageSource, /const DEFAULT_ASSISTANT_PERIOD: HistoryPeriod = "1y"/);
  assert.match(pageSource, /const DEFAULT_ASSISTANT_PRODUCT = "auto"/);
  assert.match(
    pageSource,
    /function resetAssistantFilters\(\) \{[\s\S]*?setAssistantScope\(DEFAULT_ASSISTANT_SCOPE\);[\s\S]*?setAssistantPeriod\(DEFAULT_ASSISTANT_PERIOD\);[\s\S]*?setAssistantProduct\(DEFAULT_ASSISTANT_PRODUCT\);[\s\S]*?setAssistantFiltersDirty\(false\);[\s\S]*?\}/,
  );

  const newChatStart = pageSource.indexOf("async function startNewChat()");
  const clearStart = pageSource.indexOf("async function clearConversationContext()", newChatStart);
  const submitStart = pageSource.indexOf("async function submitChat", clearStart);
  const newChatBody = pageSource.slice(newChatStart, clearStart);
  const clearBody = pageSource.slice(clearStart, submitStart);

  assert.ok(newChatStart >= 0 && clearStart > newChatStart && submitStart > clearStart);
  assert.match(newChatBody, /invalidateChatRequest\(\);\s*resetAssistantFilters\(\);/);
  assert.ok(clearBody.indexOf("resetAssistantFilters();") > clearBody.indexOf("await clearRagSession"));
  assert.ok(clearBody.indexOf("resetAssistantFilters();") < clearBody.indexOf("setConversation([])"));
});
