from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from psycopg.types.json import Jsonb

from .database import RagDatabasePool
from .models import Evidence, QueryRoute, RagV2ChatResponse, SessionState
from .settings import RagV2Settings


TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,160}$")


class SessionNotFound(RuntimeError):
    pass


class SessionExpired(RuntimeError):
    pass


class SessionAccessDenied(RuntimeError):
    pass


class SessionConflict(RuntimeError):
    pass


@dataclass(slots=True)
class SessionData:
    token: str
    internal_id: uuid.UUID
    created_at: datetime
    expires_at: datetime
    state: SessionState
    version: int


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _owner_digest(owner_token: str | None) -> str | None:
    if not owner_token:
        return None
    cleaned = owner_token.strip()
    if len(cleaned) < 16 or len(cleaned) > 256:
        raise SessionAccessDenied("Invalid session owner token")
    return _digest(cleaned)


def inherited_context_for_route(route: QueryRoute) -> dict[str, Any]:
    payload = route.model_dump(mode="json")
    values = {
        "banks": payload["banks"],
        "product_types": payload["product_types"],
        "scope": payload["scope"],
        "year": payload["year"],
        "date_range": {
            "from": payload["date_from"],
            "to": payload["date_to"],
        },
        "date_from": payload["date_from"],
        "date_to": payload["date_to"],
        "offer_ids": payload["offer_ids"],
        "field_types": payload["field_types"],
    }
    return {
        name: values[name]
        for name in route.inherited_fields
        if name in values
    }


class SessionStore:
    def __init__(
        self,
        pool: RagDatabasePool,
        settings: RagV2Settings,
    ) -> None:
        self.pool = pool
        self.settings = settings
        self._conversation_status_supported: bool | None = None

    def _expires_at(self) -> datetime:
        return datetime.now(timezone.utc) + timedelta(
            seconds=self.settings.session_ttl_seconds
        )

    def create(self, owner_token: str | None = None) -> SessionData:
        token = secrets.token_urlsafe(32)
        token_hash = _digest(token)
        owner_hash = _owner_digest(owner_token)
        internal_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        expires_at = self._expires_at()
        state = SessionState()
        with self.pool.connection() as connection:
            with connection.transaction():
                connection.execute(
                    """
                    INSERT INTO rag_sessions (
                        id, token_hash, owner_hash, created_at,
                        last_accessed_at, expires_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        internal_id,
                        token_hash,
                        owner_hash,
                        now,
                        now,
                        expires_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO rag_session_state (
                        session_id, state, version, updated_at
                    ) VALUES (%s, %s, 1, %s)
                    """,
                    (internal_id, Jsonb(state.model_dump(mode="json")), now),
                )
        return SessionData(
            token=token,
            internal_id=internal_id,
            created_at=now,
            expires_at=expires_at,
            state=state,
            version=1,
        )

    def _validate_token(self, token: str) -> str:
        if not TOKEN_PATTERN.fullmatch(token or ""):
            raise SessionNotFound("Session was not found")
        return _digest(token)

    def get(
        self,
        token: str,
        owner_token: str | None = None,
        *,
        refresh_ttl: bool = True,
    ) -> SessionData:
        token_hash = self._validate_token(token)
        supplied_owner_hash = _owner_digest(owner_token)
        with self.pool.connection() as connection:
            with connection.transaction():
                row = connection.execute(
                    """
                    SELECT
                        sessions.id,
                        sessions.owner_hash,
                        sessions.created_at,
                        sessions.expires_at,
                        state.state,
                        state.version
                    FROM rag_sessions AS sessions
                    JOIN rag_session_state AS state
                      ON state.session_id = sessions.id
                    WHERE sessions.token_hash = %s
                      AND sessions.revoked_at IS NULL
                    FOR UPDATE OF sessions, state
                    """,
                    (token_hash,),
                ).fetchone()
                if row is None:
                    raise SessionNotFound("Session was not found")
                stored_owner_hash = row["owner_hash"]
                if stored_owner_hash is not None and (
                    supplied_owner_hash is None
                    or not hmac.compare_digest(
                        str(stored_owner_hash), supplied_owner_hash
                    )
                ):
                    raise SessionAccessDenied("Session access was denied")
                now = datetime.now(timezone.utc)
                if row["expires_at"] <= now:
                    raise SessionExpired("Session has expired")
                expires_at = row["expires_at"]
                if refresh_ttl:
                    expires_at = self._expires_at()
                    connection.execute(
                        """
                        UPDATE rag_sessions
                        SET last_accessed_at = %s, expires_at = %s
                        WHERE id = %s
                        """,
                        (now, expires_at, row["id"]),
                    )
        return SessionData(
            token=token,
            internal_id=row["id"],
            created_at=row["created_at"],
            expires_at=expires_at,
            state=SessionState.model_validate(row["state"]),
            version=int(row["version"]),
        )

    def recent_user_messages(self, session_id: uuid.UUID) -> list[str]:
        limit = self.settings.history_turns
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT content
                FROM rag_messages
                WHERE session_id = %s AND role = 'user'
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (session_id, limit),
            ).fetchall()
        return [str(row["content"]) for row in reversed(rows)]

    def recent_conversation_messages(
        self,
        session_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                WITH recent_turns AS (
                    SELECT turn_id, MAX(id) AS last_message_id
                    FROM rag_messages
                    WHERE session_id = %s
                    GROUP BY turn_id
                    ORDER BY MAX(id) DESC
                    LIMIT %s
                )
                SELECT
                    messages.role,
                    messages.content,
                    messages.status,
                    messages.route,
                    messages.inherited_context,
                    messages.id
                FROM rag_messages AS messages
                JOIN recent_turns
                  ON recent_turns.turn_id = messages.turn_id
                WHERE messages.session_id = %s
                ORDER BY messages.id ASC
                """,
                (
                    session_id,
                    self.settings.history_turns,
                    session_id,
                ),
            ).fetchall()

        budget = self.settings.history_max_chars
        per_message_limit = max(
            500,
            budget // max(1, self.settings.history_turns * 2),
        )
        selected: list[dict[str, Any]] = []
        for row in reversed(rows):
            if budget <= 0:
                break
            content = str(row["content"]).strip()
            if len(content) > per_message_limit:
                content = content[: per_message_limit - 3].rstrip() + "..."
            if len(content) > budget:
                content = content[:budget].rstrip()
            if not content:
                continue
            selected.append(
                {
                    "role": str(row["role"]),
                    "content": content,
                    "status": str(row["status"]) if row["status"] else None,
                    "route": row["route"] if row["route"] else None,
                    "inherited_context": (
                        row["inherited_context"]
                        if row["inherited_context"]
                        else {}
                    ),
                }
            )
            budget -= len(content)
        return list(reversed(selected))

    def recent_verified_evidence_refs(
        self,
        session_id: uuid.UUID,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    evidence.turn_id,
                    evidence.source_id,
                    evidence.chunk_id,
                    evidence.offer_id,
                    evidence.document_id
                FROM rag_turn_evidence AS evidence
                JOIN rag_messages AS messages
                  ON messages.id = evidence.assistant_message_id
                 AND messages.session_id = evidence.session_id
                WHERE evidence.session_id = %s
                  AND messages.role = 'assistant'
                  AND messages.status = 'verified'
                  AND POSITION(
                      ('[' || evidence.source_id || ']') IN messages.content
                  ) > 0
                ORDER BY evidence.assistant_message_id DESC, evidence.source_id
                LIMIT %s
                """,
                (session_id, limit),
            ).fetchall()
        return [
            {
                "turn_id": row["turn_id"],
                "source_id": str(row["source_id"]),
                "chunk_id": str(row["chunk_id"]),
                "offer_id": str(row["offer_id"]),
                "document_id": str(row["document_id"]),
            }
            for row in rows
            if row["chunk_id"] and row["offer_id"] and row["document_id"]
        ]

    def conversation_transcript(
        self,
        session: SessionData,
    ) -> list[RagV2ChatResponse]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                WITH recent_turns AS (
                    SELECT turn_id, MAX(id) AS last_message_id
                    FROM rag_messages
                    WHERE session_id = %s
                    GROUP BY turn_id
                    ORDER BY MAX(id) DESC
                    LIMIT %s
                )
                SELECT
                    messages.turn_id,
                    messages.role,
                    messages.content,
                    messages.status,
                    messages.route,
                    messages.inherited_context,
                    messages.created_at,
                    messages.id
                FROM rag_messages AS messages
                JOIN recent_turns
                  ON recent_turns.turn_id = messages.turn_id
                WHERE messages.session_id = %s
                ORDER BY messages.id ASC
                """,
                (
                    session.internal_id,
                    self.settings.transcript_turns,
                    session.internal_id,
                ),
            ).fetchall()
            turn_ids = list(dict.fromkeys(row["turn_id"] for row in rows))
            evidence_rows = []
            if turn_ids:
                evidence_rows = connection.execute(
                    """
                    SELECT turn_id, source_id, evidence
                    FROM rag_turn_evidence
                    WHERE session_id = %s
                      AND turn_id = ANY(%s)
                    ORDER BY assistant_message_id ASC, source_id ASC
                    """,
                    (session.internal_id, turn_ids),
                ).fetchall()

        evidence_by_turn: dict[uuid.UUID, list[Evidence]] = {}
        for row in evidence_rows:
            try:
                parsed = Evidence.model_validate(row["evidence"])
            except ValueError:
                continue
            evidence_by_turn.setdefault(row["turn_id"], []).append(parsed)

        turns: dict[uuid.UUID, dict[str, Any]] = {}
        for row in rows:
            item = turns.setdefault(
                row["turn_id"],
                {
                    "query": "",
                    "answer": "",
                    "status": None,
                    "route": None,
                    "inherited_context": {},
                },
            )
            role = str(row["role"])
            if role == "user":
                item["query"] = str(row["content"])
                continue
            if role != "assistant":
                continue
            item["answer"] = str(row["content"])
            item["status"] = str(row["status"]) if row["status"] else None
            item["route"] = row["route"]
            item["inherited_context"] = row["inherited_context"] or {}

        output: list[RagV2ChatResponse] = []
        for turn_id in turn_ids:
            item = turns.get(turn_id) or {}
            if not item.get("query") or not item.get("answer"):
                continue
            try:
                route = QueryRoute.model_validate(item.get("route") or {})
            except ValueError:
                continue
            status = item.get("status")
            if status is None and route.intent == "chat":
                status = "conversational"
            if status not in {
                "verified",
                "rejected",
                "insufficient_evidence",
                "needs_clarification",
                "conversational",
            }:
                continue
            output.append(
                RagV2ChatResponse(
                    session_id=session.token,
                    query=item["query"],
                    standalone_query=route.standalone_query,
                    answer=item["answer"],
                    status=status,
                    inherited_context=item["inherited_context"],
                    route=route,
                    evidence=evidence_by_turn.get(turn_id, []),
                    issues=[],
                    diagnostics={"restored_from_session": True},
                )
            )
        return output

    def add_user_message(
        self,
        session_id: uuid.UUID,
        content: str,
    ) -> tuple[uuid.UUID, int]:
        turn_id = uuid.uuid4()
        with self.pool.connection() as connection:
            with connection.transaction():
                row = connection.execute(
                    """
                    INSERT INTO rag_messages (
                        session_id, turn_id, role, content
                    ) VALUES (%s, %s, 'user', %s)
                    RETURNING id
                    """,
                    (session_id, turn_id, content),
                ).fetchone()
        return turn_id, int(row["id"])

    def finish_turn(
        self,
        session: SessionData,
        turn_id: uuid.UUID,
        answer: str,
        status: str,
        route: QueryRoute,
        evidence: list[Evidence],
        new_state: SessionState,
        *,
        user_content: str | None = None,
    ) -> int:
        now = datetime.now(timezone.utc)
        route_payload = route.model_dump(mode="json")
        inherited_payload = inherited_context_for_route(route)
        stored_status: str | None = status
        if status == "conversational":
            if self._conversation_status_supported is None:
                with self.pool.connection() as connection:
                    row = connection.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM pg_constraint
                            WHERE conrelid = 'public.rag_messages'::REGCLASS
                              AND contype = 'c'
                              AND pg_get_constraintdef(oid)
                                  ILIKE '%conversational%'
                        ) AS supported
                        """
                    ).fetchone()
                self._conversation_status_supported = bool(
                    row and row["supported"]
                )
            if not self._conversation_status_supported:
                stored_status = None
        with self.pool.connection() as connection:
            with connection.transaction():
                if user_content is not None:
                    connection.execute(
                        """
                        INSERT INTO rag_messages (
                            session_id, turn_id, role, content
                        ) VALUES (%s, %s, 'user', %s)
                        """,
                        (session.internal_id, turn_id, user_content),
                    )
                row = connection.execute(
                    """
                    INSERT INTO rag_messages (
                        session_id, turn_id, role, content, route,
                        inherited_context, status
                    ) VALUES (%s, %s, 'assistant', %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        session.internal_id,
                        turn_id,
                        answer,
                        Jsonb(route_payload),
                        Jsonb(inherited_payload),
                        stored_status,
                    ),
                ).fetchone()
                message_id = int(row["id"])
                for item in evidence:
                    connection.execute(
                        """
                        INSERT INTO rag_turn_evidence (
                            session_id, turn_id, assistant_message_id, source_id,
                            chunk_id, offer_id, document_id, evidence
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            session.internal_id,
                            turn_id,
                            message_id,
                            item.source_id,
                            item.chunk_id,
                            item.offer_id,
                            item.document_id,
                            Jsonb(item.model_dump(mode="json")),
                        ),
                    )
                updated = connection.execute(
                    """
                    UPDATE rag_session_state
                    SET state = %s, version = version + 1, updated_at = %s
                    WHERE session_id = %s AND version = %s
                    RETURNING version
                    """,
                    (
                        Jsonb(new_state.model_dump(mode="json")),
                        now,
                        session.internal_id,
                        session.version,
                    ),
                ).fetchone()
                if updated is None:
                    raise SessionConflict("Session state changed concurrently")
        return message_id

    def clear(
        self,
        token: str,
        owner_token: str | None = None,
    ) -> SessionData:
        session = self.get(token, owner_token)
        state = SessionState()
        now = datetime.now(timezone.utc)
        with self.pool.connection() as connection:
            with connection.transaction():
                connection.execute(
                    "DELETE FROM rag_turn_evidence WHERE session_id = %s",
                    (session.internal_id,),
                )
                connection.execute(
                    "DELETE FROM rag_messages WHERE session_id = %s",
                    (session.internal_id,),
                )
                row = connection.execute(
                    """
                    UPDATE rag_session_state
                    SET state = %s, version = version + 1, updated_at = %s
                    WHERE session_id = %s
                    RETURNING version
                    """,
                    (
                        Jsonb(state.model_dump(mode="json")),
                        now,
                        session.internal_id,
                    ),
                ).fetchone()
        session.state = state
        session.version = int(row["version"])
        return session

    def revoke(
        self,
        token: str,
        owner_token: str | None = None,
    ) -> None:
        session = self.get(token, owner_token, refresh_ttl=False)
        with self.pool.connection() as connection:
            with connection.transaction():
                connection.execute(
                    """
                    UPDATE rag_sessions
                    SET revoked_at = %s
                    WHERE id = %s
                    """,
                    (datetime.now(timezone.utc), session.internal_id),
                )
