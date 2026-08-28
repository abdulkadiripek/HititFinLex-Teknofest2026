from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request

from .models import (
    RagV2ChatRequest,
    RagV2ChatResponse,
    SessionClearResponse,
    SessionDeleteResponse,
    SessionResponse,
    SessionTranscriptResponse,
)
from .service import RagV2Service
from .sessions import (
    SessionAccessDenied,
    SessionConflict,
    SessionExpired,
    SessionNotFound,
)


logger = logging.getLogger("hititfinlex.rag_v2")
router = APIRouter(prefix="/rag/v2", tags=["RAG V2"])


def _service(request: Request) -> RagV2Service:
    service = getattr(request.app.state, "rag_v2_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG V2 is not ready.")
    return service


def _session_error(error: Exception) -> HTTPException:
    if isinstance(error, SessionExpired):
        return HTTPException(status_code=410, detail="Session has expired.")
    if isinstance(error, (SessionNotFound, SessionAccessDenied)):
        return HTTPException(status_code=404, detail="Session was not found.")
    if isinstance(error, SessionConflict):
        return HTTPException(status_code=409, detail="Session changed concurrently.")
    return HTTPException(status_code=500, detail="Session operation failed.")


@router.post("/sessions", response_model=SessionResponse)
def create_session(
    request: Request,
    client_id: str | None = Header(default=None, alias="X-RAG-Client-Id"),
):
    service = _service(request)
    try:
        session = service.sessions.create(client_id)
    except SessionAccessDenied as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid client identifier.",
        ) from error
    except Exception as error:
        logger.exception("RAG V2 session creation failed")
        raise HTTPException(
            status_code=500, detail="Session creation failed."
        ) from error
    return SessionResponse(
        session_id=session.token,
        created_at=session.created_at,
        expires_at=session.expires_at,
        state=session.state,
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: str,
    request: Request,
    client_id: str | None = Header(default=None, alias="X-RAG-Client-Id"),
):
    try:
        session = _service(request).sessions.get(session_id, client_id)
    except (SessionNotFound, SessionExpired, SessionAccessDenied) as error:
        raise _session_error(error) from error
    return SessionResponse(
        session_id=session.token,
        created_at=session.created_at,
        expires_at=session.expires_at,
        state=session.state,
    )


@router.get("/session", response_model=SessionResponse)
def get_session_by_header(
    request: Request,
    session_id: str = Header(alias="X-RAG-Session-Id"),
    client_id: str | None = Header(default=None, alias="X-RAG-Client-Id"),
):
    return get_session(session_id, request, client_id)


@router.get(
    "/sessions/{session_id}/messages",
    response_model=SessionTranscriptResponse,
)
def get_session_messages(
    session_id: str,
    request: Request,
    client_id: str | None = Header(default=None, alias="X-RAG-Client-Id"),
):
    try:
        service = _service(request)
        session = service.sessions.get(session_id, client_id)
        messages = service.sessions.conversation_transcript(session)
    except (SessionNotFound, SessionExpired, SessionAccessDenied) as error:
        raise _session_error(error) from error
    return SessionTranscriptResponse(
        session_id=session.token,
        expires_at=session.expires_at,
        messages=messages,
    )


@router.get(
    "/session/messages",
    response_model=SessionTranscriptResponse,
)
def get_session_messages_by_header(
    request: Request,
    session_id: str = Header(alias="X-RAG-Session-Id"),
    client_id: str | None = Header(default=None, alias="X-RAG-Client-Id"),
):
    return get_session_messages(session_id, request, client_id)


@router.post(
    "/sessions/{session_id}/clear",
    response_model=SessionClearResponse,
)
def clear_session(
    session_id: str,
    request: Request,
    client_id: str | None = Header(default=None, alias="X-RAG-Client-Id"),
):
    try:
        session = _service(request).sessions.clear(session_id, client_id)
    except (SessionNotFound, SessionExpired, SessionAccessDenied) as error:
        raise _session_error(error) from error
    return SessionClearResponse(
        session_id=session.token,
        cleared=True,
        expires_at=session.expires_at,
        state=session.state,
    )


@router.post("/session/clear", response_model=SessionClearResponse)
def clear_session_by_header(
    request: Request,
    session_id: str = Header(alias="X-RAG-Session-Id"),
    client_id: str | None = Header(default=None, alias="X-RAG-Client-Id"),
):
    return clear_session(session_id, request, client_id)


@router.delete(
    "/sessions/{session_id}",
    response_model=SessionDeleteResponse,
)
def delete_session(
    session_id: str,
    request: Request,
    client_id: str | None = Header(default=None, alias="X-RAG-Client-Id"),
):
    try:
        _service(request).sessions.revoke(session_id, client_id)
    except (SessionNotFound, SessionExpired, SessionAccessDenied) as error:
        raise _session_error(error) from error
    return SessionDeleteResponse(deleted=True)


@router.delete("/session", response_model=SessionDeleteResponse)
def delete_session_by_header(
    request: Request,
    session_id: str = Header(alias="X-RAG-Session-Id"),
    client_id: str | None = Header(default=None, alias="X-RAG-Client-Id"),
):
    return delete_session(session_id, request, client_id)


@router.post("/chat", response_model=RagV2ChatResponse)
def rag_v2_chat(
    payload: RagV2ChatRequest,
    request: Request,
    client_id: str | None = Header(default=None, alias="X-RAG-Client-Id"),
):
    service = _service(request)
    try:
        return service.chat(payload, client_id)
    except (
        SessionNotFound,
        SessionExpired,
        SessionAccessDenied,
        SessionConflict,
    ) as error:
        raise _session_error(error) from error
    except Exception as error:
        logger.exception("RAG V2 chat failed")
        raise HTTPException(
            status_code=500,
            detail="RAG V2 request failed.",
        ) from error
