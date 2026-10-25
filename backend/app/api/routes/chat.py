"""Text chat and conversation session management routes."""

import json
import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse, MetricsResponse, SessionDeleteResponse, SessionSummary
from app.services.llm import OllamaModelMissingError, OllamaResponseError, OllamaUnavailableError
from app.services.session_manager import SessionCapacityError

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def _service_error(error: Exception) -> HTTPException:
    if isinstance(error, OllamaModelMissingError):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error))
    if isinstance(error, OllamaUnavailableError):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error))
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error))


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    try:
        result = await request.app.state.pipeline.chat(payload.message, payload.session_id, payload.language)
    except SessionCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except (OllamaUnavailableError, OllamaModelMissingError, OllamaResponseError) as exc:
        raise _service_error(exc) from exc
    return ChatResponse(
        session_id=result.session_id,
        response=result.response,
        metrics=MetricsResponse(**result.metrics),
        created_at=result.created_at,
    )


@router.post("/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    async def events():
        try:
            async for event in request.app.state.pipeline.chat_stream(
                payload.message, payload.session_id, payload.language
            ):
                if await request.is_disconnected():
                    return
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except SessionCapacityError as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
        except (OllamaUnavailableError, OllamaModelMissingError, OllamaResponseError) as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
        except Exception:
            logger.exception("Unexpected streaming text failure")
            yield json.dumps({"type": "error", "detail": "The local text pipeline failed."}) + "\n"
    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(request: Request) -> list[SessionSummary]:
    sessions = await request.app.state.sessions.list_sessions()
    return [SessionSummary(session_id=sid, message_count=count, updated_at=updated) for sid, count, updated in sessions]


@router.delete("/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, request: Request) -> SessionDeleteResponse:
    if len(session_id) > 128:
        raise HTTPException(status_code=422, detail="Session ID must be 128 characters or fewer.")
    deleted = await request.app.state.sessions.delete(session_id)
    return SessionDeleteResponse(session_id=session_id, deleted=deleted)
