export const RAG_SESSION_STORAGE_KEY = "hititfinlex.rag.v2.session";
export const RAG_CLIENT_STORAGE_KEY = "hititfinlex.rag.v2.client";
export const SAFE_FALLBACK_ANSWER = "Yeterli doğrulanabilir kaynak bulunamadı.";
export const DEFAULT_CLARIFICATION = "Sorunuzu biraz daha açık belirtir misiniz?";

export type RagV2Status =
  | "verified"
  | "rejected"
  | "insufficient_evidence"
  | "needs_clarification"
  | "conversational";

export type RagV2Scope = "current" | "historical" | "all";

export type RagV2Route = {
  standalone_query?: string;
  intent?: "lookup" | "compare" | "list" | "calculate" | "historical" | "clarification" | "chat";
  banks?: string[];
  product_types?: string[];
  field_types?: string[];
  scope?: RagV2Scope;
  year?: number | null;
  date_from?: string | null;
  date_to?: string | null;
  offer_ids?: string[];
  inherited_fields?: string[];
  needs_clarification?: boolean;
  clarification_question?: string | null;
};

export type RagEvidence = {
  source_id: string | number;
  document_id?: string | number | null;
  offer_id?: string | null;
  chunk_id?: string | null;
  bank_name?: string | null;
  page_title?: string | null;
  title?: string | null;
  source_url?: string | null;
  canonical_url?: string | null;
  archive_url?: string | null;
  snapshot_date?: string | null;
  effective_date?: string | null;
  product_type?: string | null;
  product_types?: string[];
  content?: string | null;
  evidence_text?: string | null;
  snippet?: string | null;
  dense_score?: number | null;
  semantic_score?: number | null;
  lexical_score?: number | null;
  rrf_score?: number | null;
  hybrid_score?: number | null;
  classification_confidence?: number | null;
  classification_status?: string | null;
  verified?: boolean;
};

export type RagV2Response = {
  session_id: string;
  query: string;
  standalone_query: string;
  answer: string;
  status: RagV2Status;
  inherited_context: Record<string, unknown>;
  route: RagV2Route;
  evidence: RagEvidence[];
  issues: string[];
  diagnostics: Record<string, unknown>;
};

export type RagV2Request = {
  session_id: string | null;
  query: string;
  top_k: number;
  use_reranker: false;
  scope?: RagV2Scope;
  date_from?: string | null;
  date_to?: string | null;
  product_types?: string[];
};

export type RagSessionResponse = {
  session_id: string;
  expires_at?: string | null;
  state?: Record<string, unknown>;
};

export type RagSessionMessagesResponse = {
  session_id: string;
  expires_at?: string | null;
  messages: RagV2Response[];
};

export type RagChatRequestOptions = {
  fetcher: typeof fetch;
  apiBaseUrl: string;
  clientId: string;
  sessionId: string | null;
  request: Omit<RagV2Request, "session_id">;
  signal?: AbortSignal;
};

export type RagChatRequestResult = {
  response: RagV2Response;
  retriedWithoutExpiredSession: boolean;
};

export type ContextChip = {
  kind: "bank" | "product" | "year" | "scope";
  label: string;
  value: string;
};

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export class RagApiError extends Error {
  readonly status: number;

  constructor(status: number) {
    super(`RAG API request failed with status ${status}`);
    this.name = "RagApiError";
    this.status = status;
  }
}

const VALID_STATUS = new Set<RagV2Status>([
  "verified",
  "rejected",
  "insufficient_evidence",
  "needs_clarification",
  "conversational",
]);

const OPAQUE_ID_PATTERN = /^[A-Za-z0-9._~-]{16,512}$/;
const SOURCE_ID_PATTERN = /^S([1-9]\d*)$/i;

export function isOpaqueIdentifier(value: unknown): value is string {
  return typeof value === "string" && OPAQUE_ID_PATTERN.test(value);
}

export function createClientId(randomUuid?: () => string): string {
  const uuid = randomUuid ? randomUuid() : globalThis.crypto.randomUUID();
  const clientId = `rag-web-${uuid}`;
  if (!isOpaqueIdentifier(clientId)) {
    throw new Error("Secure client id generation failed");
  }
  return clientId;
}

export function getOrCreateClientId(
  storage: StorageLike,
  randomUuid?: () => string,
): string {
  const stored = storage.getItem(RAG_CLIENT_STORAGE_KEY);
  if (isOpaqueIdentifier(stored)) return stored;
  storage.removeItem(RAG_CLIENT_STORAGE_KEY);
  const clientId = createClientId(randomUuid);
  storage.setItem(RAG_CLIENT_STORAGE_KEY, clientId);
  return clientId;
}

export function loadSessionId(storage: StorageLike): string | null {
  const stored = storage.getItem(RAG_SESSION_STORAGE_KEY);
  if (isOpaqueIdentifier(stored)) return stored;
  if (stored !== null) storage.removeItem(RAG_SESSION_STORAGE_KEY);
  return null;
}

export function saveSessionId(storage: StorageLike, sessionId: string): void {
  if (!isOpaqueIdentifier(sessionId)) {
    throw new Error("Invalid opaque session id");
  }
  storage.setItem(RAG_SESSION_STORAGE_KEY, sessionId);
}

export function clearSessionId(storage: StorageLike): void {
  storage.removeItem(RAG_SESSION_STORAGE_KEY);
}

export function initializePersistentRagStorage(
  persistentStorage: StorageLike,
  legacyStorage: StorageLike,
  randomUuid?: () => string,
): { clientId: string; sessionId: string | null } {
  const persistentClient = persistentStorage.getItem(RAG_CLIENT_STORAGE_KEY);
  const persistentSession = persistentStorage.getItem(RAG_SESSION_STORAGE_KEY);
  const legacyClient = legacyStorage.getItem(RAG_CLIENT_STORAGE_KEY);
  const legacySession = legacyStorage.getItem(RAG_SESSION_STORAGE_KEY);
  const persistentPairValid = isOpaqueIdentifier(persistentClient)
    && isOpaqueIdentifier(persistentSession);
  const legacyPairValid = isOpaqueIdentifier(legacyClient)
    && isOpaqueIdentifier(legacySession);

  let clientId: string;
  let sessionId: string | null;
  if (persistentPairValid) {
    clientId = persistentClient;
    sessionId = persistentSession;
  } else if (legacyPairValid) {
    clientId = legacyClient;
    sessionId = legacySession;
  } else {
    clientId = isOpaqueIdentifier(persistentClient)
      ? persistentClient
      : isOpaqueIdentifier(legacyClient)
        ? legacyClient
        : createClientId(randomUuid);
    sessionId = null;
  }

  persistentStorage.setItem(RAG_CLIENT_STORAGE_KEY, clientId);
  if (sessionId) persistentStorage.setItem(RAG_SESSION_STORAGE_KEY, sessionId);
  else persistentStorage.removeItem(RAG_SESSION_STORAGE_KEY);
  legacyStorage.removeItem(RAG_CLIENT_STORAGE_KEY);
  legacyStorage.removeItem(RAG_SESSION_STORAGE_KEY);
  return { clientId, sessionId };
}

export function clientHeaders(clientId: string): Record<string, string> {
  if (!isOpaqueIdentifier(clientId)) {
    throw new Error("Invalid client id");
  }
  return {
    "Content-Type": "application/json",
    "X-RAG-Client-Id": clientId,
  };
}

export async function requestRagJson<T>(
  fetcher: typeof fetch,
  input: string,
  init: RequestInit,
): Promise<T> {
  const response = await fetcher(input, init);
  if (!response.ok) throw new RagApiError(response.status);
  return (await response.json()) as T;
}

export function isExpiredSessionError(error: unknown): boolean {
  return error instanceof RagApiError && (error.status === 404 || error.status === 410);
}

export async function requestRagChatWithRetry(
  options: RagChatRequestOptions,
): Promise<RagChatRequestResult> {
  let activeSessionId = options.sessionId;
  let retriedWithoutExpiredSession = false;

  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const raw = await requestRagJson<unknown>(
        options.fetcher,
        apiUrl(options.apiBaseUrl, "/rag/v2/chat"),
        {
          method: "POST",
          headers: clientHeaders(options.clientId),
          body: JSON.stringify({
            session_id: activeSessionId,
            query: options.request.query,
            top_k: options.request.top_k,
            use_reranker: false,
            ...(options.request.scope ? { scope: options.request.scope } : {}),
            ...(options.request.date_from !== undefined ? { date_from: options.request.date_from } : {}),
            ...(options.request.date_to !== undefined ? { date_to: options.request.date_to } : {}),
            ...(options.request.product_types ? { product_types: options.request.product_types } : {}),
          } satisfies RagV2Request),
          signal: options.signal,
        },
      );
      const response = parseRagV2Response(raw);
      if (!response) throw new Error("Invalid RAG V2 response");
      return { response, retriedWithoutExpiredSession };
    } catch (error) {
      if (attempt === 0 && activeSessionId && isExpiredSessionError(error)) {
        activeSessionId = null;
        retriedWithoutExpiredSession = true;
        continue;
      }
      throw error;
    }
  }

  throw new Error("RAG V2 retry exhausted");
}

export function parseRagV2Response(value: unknown): RagV2Response | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const candidate = value as Record<string, unknown>;
  if (!isOpaqueIdentifier(candidate.session_id)) return null;
  if (typeof candidate.query !== "string") return null;
  if (typeof candidate.standalone_query !== "string") return null;
  if (typeof candidate.answer !== "string") return null;
  if (!VALID_STATUS.has(candidate.status as RagV2Status)) return null;
  if (!Array.isArray(candidate.evidence) || !Array.isArray(candidate.issues)) return null;
  if (!candidate.issues.every((issue) => typeof issue === "string")) return null;
  if (!isRecord(candidate.route) || !isRecord(candidate.inherited_context) || !isRecord(candidate.diagnostics)) return null;
  const evidence = candidate.evidence.filter(isEvidence);
  if (evidence.length !== candidate.evidence.length) return null;
  const sourceIds = evidence.map((source, index) => evidenceSourceId(source, index));
  if (new Set(sourceIds).size !== sourceIds.length) return null;
  if (candidate.status === "needs_clarification" && candidate.route.needs_clarification !== true) return null;
  if (
    candidate.status === "conversational"
    && (
      candidate.route.intent !== "chat"
      || candidate.route.needs_clarification !== false
      || evidence.length > 0
      || !candidate.answer.trim()
      || /\[S[1-9]\d*\]/i.test(candidate.answer)
    )
  ) {
    return null;
  }
  return {
    session_id: candidate.session_id,
    query: candidate.query,
    standalone_query: candidate.standalone_query,
    answer: candidate.answer,
    status: candidate.status as RagV2Status,
    inherited_context: candidate.inherited_context,
    route: candidate.route as RagV2Route,
    evidence,
    issues: candidate.issues,
    diagnostics: candidate.diagnostics,
  };
}

export function parseRagSessionMessagesResponse(
  value: unknown,
): RagSessionMessagesResponse | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const candidate = value as Record<string, unknown>;
  if (!isOpaqueIdentifier(candidate.session_id) || !Array.isArray(candidate.messages)) return null;
  if (
    candidate.expires_at !== undefined
    && candidate.expires_at !== null
    && typeof candidate.expires_at !== "string"
  ) {
    return null;
  }
  const messages = candidate.messages.map(parseRagV2Response);
  if (messages.some((message) => message === null)) return null;
  return {
    session_id: candidate.session_id,
    expires_at: candidate.expires_at as string | null | undefined,
    messages: messages as RagV2Response[],
  };
}

export function safeHttpUrl(value: unknown): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return undefined;
    if (parsed.username || parsed.password) return undefined;
    return parsed.href;
  } catch {
    return undefined;
  }
}

export function evidenceSourceId(source: RagEvidence, index = 0): string {
  const raw = String(source.source_id).trim();
  const match = SOURCE_ID_PATTERN.exec(raw);
  if (match) return `S${match[1]}`;
  if (/^[1-9]\d*$/.test(raw)) return `S${raw}`;
  return `S${index + 1}`;
}

export function evidenceTitle(source: RagEvidence): string {
  return source.page_title?.trim() || source.title?.trim() || "Banka belgesi";
}

export function evidenceText(source: RagEvidence): string {
  return source.evidence_text?.trim() || source.content?.trim() || source.snippet?.trim() || "";
}

export function evidenceUrl(source: RagEvidence): string | undefined {
  return safeHttpUrl(source.archive_url ?? source.canonical_url ?? source.source_url);
}

export function evidenceScore(source: RagEvidence): number | null {
  for (const value of [source.rrf_score, source.hybrid_score, source.dense_score, source.semantic_score]) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

export function answerTokens(text: string): string[] {
  return text.split(/(\*\*[^*]+\*\*|\[S[1-9]\d*\])/gi).filter(Boolean);
}

export function knownCitationIds(evidence: RagEvidence[]): Set<string> {
  return new Set(evidence.map((source, index) => evidenceSourceId(source, index)));
}

export function citationIdFromToken(token: string): string | null {
  const match = /^\[(S[1-9]\d*)\]$/i.exec(token);
  return match ? match[1].toUpperCase() : null;
}

export function messageSourceAnchor(messageId: number, sourceId: string): string {
  const normalized = SOURCE_ID_PATTERN.test(sourceId) ? sourceId.toUpperCase() : "S0";
  return `message-${messageId}-source-${normalized}`;
}

export function displayAnswer(response: RagV2Response): string {
  if (response.status === "needs_clarification") {
    return response.route.clarification_question?.trim() || response.answer.trim() || DEFAULT_CLARIFICATION;
  }
  if (response.status === "conversational") {
    return conversationalDisplayAnswer(response) ?? SAFE_FALLBACK_ANSWER;
  }
  return verifiedDisplayAnswer(response) ?? SAFE_FALLBACK_ANSWER;
}

export function responseStatusForDisplay(response: RagV2Response): RagV2Status {
  if (response.status === "verified" && verifiedDisplayAnswer(response) === null) {
    return "rejected";
  }
  if (response.status === "conversational" && conversationalDisplayAnswer(response) === null) {
    return "rejected";
  }
  return response.status;
}

function conversationalDisplayAnswer(response: RagV2Response): string | null {
  if (
    response.status !== "conversational"
    || response.route.intent !== "chat"
    || response.route.needs_clarification !== false
    || response.evidence.length > 0
    || !response.answer.trim()
    || /\[S[1-9]\d*\]/i.test(response.answer)
  ) {
    return null;
  }
  return response.answer.trim();
}

function verifiedDisplayAnswer(response: RagV2Response): string | null {
  if (response.status !== "verified" || response.evidence.length === 0 || !response.answer.trim()) return null;
  const knownSources = knownCitationIds(response.evidence);
  const citationTokens = [...response.answer.matchAll(/\[(S[^\]\s]*)\]/gi)]
    .map((match) => match[1].toUpperCase());
  if (
    citationTokens.length === 0
    || citationTokens.some((sourceId) => !SOURCE_ID_PATTERN.test(sourceId) || !knownSources.has(sourceId))
  ) {
    return null;
  }
  return response.answer.trim();
}

export function statusLabel(status: RagV2Status): string {
  const labels: Record<RagV2Status, string> = {
    verified: "Doğrulandı",
    rejected: "Yanıt engellendi",
    insufficient_evidence: "Kanıt yetersiz",
    needs_clarification: "Açıklama gerekli",
    conversational: "Sohbet",
  };
  return labels[status];
}

export function buildInheritedContextChips(
  context: Record<string, unknown>,
  route: RagV2Route = {},
): ContextChip[] {
  const chips: ContextChip[] = [];
  const inheritedFields = new Set(asStringArray(context.inherited_fields ?? route.inherited_fields));
  const hasFieldFilter = inheritedFields.size > 0;
  const accepts = (...names: string[]) => !hasFieldFilter || names.some((name) => inheritedFields.has(name));
  const banks = asStringArray(context.banks ?? context.active_banks);
  const products = asStringArray(context.product_types ?? context.active_products);
  const year = context.year ?? context.active_year;
  const scope = context.scope ?? context.active_scope;

  if (accepts("banks", "active_banks")) {
    for (const value of banks) chips.push({ kind: "bank", label: "Banka", value });
  }
  if (accepts("product_types", "active_products")) {
    for (const value of products) chips.push({ kind: "product", label: "Ürün", value });
  }
  if (accepts("year", "active_year") && (typeof year === "number" || typeof year === "string")) {
    chips.push({ kind: "year", label: "Yıl", value: String(year) });
  }
  if (accepts("scope", "active_scope") && typeof scope === "string") {
    const scopeLabels: Record<string, string> = { current: "Güncel", historical: "Tarihsel", all: "Tümü" };
    chips.push({ kind: "scope", label: "Kapsam", value: scopeLabels[scope] ?? scope });
  }

  return chips.filter(
    (chip, index, values) =>
      values.findIndex((candidate) => candidate.kind === chip.kind && candidate.value === chip.value) === index,
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isEvidence(value: unknown): value is RagEvidence {
  if (!isRecord(value)) return false;
  return typeof value.source_id === "string" && SOURCE_ID_PATTERN.test(value.source_id.trim());
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === "string" && Boolean(item.trim()))
    .map((item) => item.trim());
}

function apiUrl(baseUrl: string, path: string): string {
  return `${baseUrl.replace(/\/$/, "")}${path}`;
}
