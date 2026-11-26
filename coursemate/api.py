import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, StringConstraints

from coursemate.chat import Final, Token, Turn, stream_answer
from coursemate.embeddings import Embedder
from coursemate.llm import LLMError
from coursemate.rag import DEFAULT_MIN_SCORE
from coursemate.retrieval import Mode
from coursemate.sessions import SessionStore, UnknownSession
from coursemate.sse import format_sse
from coursemate.store import StoreError, VectorStore
from coursemate.streaming import StreamingLLM

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    # FastAPI checks this before the route runs: bad input gets a 422 and never reaches the model
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    session_id: str | None = None


def create_app(
    embedder: Embedder,
    store: VectorStore,
    llm: StreamingLLM,
    *,
    sessions: SessionStore | None = None,
    k: int = 4,
    min_score: float = DEFAULT_MIN_SCORE,
    mode: Mode = "vector",
) -> FastAPI:
    app = FastAPI(title="CourseMate")
    sessions = SessionStore() if sessions is None else sessions

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "provider": llm.provider, "model": llm.model}

    @app.post("/api/sessions")
    async def create_session() -> dict[str, str]:
        return {"session_id": sessions.create()}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> list[dict[str, str]]:
        try:
            turns = sessions.history(session_id)
        except UnknownSession:
            raise HTTPException(status_code=404, detail="unknown session") from None
        return [{"question": t.question, "answer": t.answer} for t in turns]

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        # Errors that happen before streaming starts can still be a real HTTP status.
        if request.session_id is None:
            session_id = sessions.create()
        elif sessions.exists(request.session_id):
            session_id = request.session_id
        else:
            raise HTTPException(status_code=404, detail="unknown session")
        history = sessions.history(session_id)
        question = request.question

        async def events() -> AsyncIterator[str]:
            yield format_sse("session", {"session_id": session_id})
            try:
                answer = stream_answer(
                    question,
                    history,
                    embedder,
                    store,
                    llm,
                    k=k,
                    min_score=min_score,
                    mode=mode,
                    session_id=session_id,
                )
                # if the browser leaves, close the answer now, so its spans end
                async with contextlib.aclosing(answer):
                    async for event in answer:
                        if isinstance(event, Token):
                            yield format_sse("token", {"text": event.text})
                        elif isinstance(event, Final):
                            # the session may have been dropped to make room while we streamed
                            with contextlib.suppress(UnknownSession):
                                sessions.add(session_id, Turn(question, event.answer))
                            yield format_sse("done", done_payload(event, llm))
            except (LLMError, StoreError) as exc:
                # the response has already started, so the error travels as an event
                yield format_sse("error", {"message": str(exc)})

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    return app


def done_payload(final: Final, llm: StreamingLLM) -> dict:
    return {
        "found": final.found,
        "answer": final.answer,
        "sources": [
            {"n": n, "source": hit.source, "heading": hit.heading, "score": hit.score}
            for n, hit in zip(final.citations, final.sources, strict=True)
        ],
        "reason": final.reason,
        "provider": llm.provider,
        "model": final.model,
        "input_tokens": final.input_tokens,
        "output_tokens": final.output_tokens,
        "first_token_ms": final.first_token_ms,
        "total_ms": final.total_ms,
        # None (cost unknown) when the model isn't in the price table; the page shows that as such
        "cost_usd": final.cost_usd,
        "cached": final.cached,
    }
