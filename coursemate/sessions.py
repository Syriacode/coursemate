import uuid

from coursemate.chat import Turn


class UnknownSession(KeyError):
    """No session with this id (never created, or dropped to make room)."""


class SessionStore:
    """Conversation history kept in memory, so it is lost when the server restarts."""

    def __init__(self, max_turns: int = 5, max_sessions: int = 1000) -> None:
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        # a dict keeps insertion order, so the first key is the oldest session
        self._sessions: dict[str, list[Turn]] = {}

    def create(self) -> str:
        if len(self._sessions) >= self.max_sessions:
            del self._sessions[next(iter(self._sessions))]  # drop the oldest
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = []
        return session_id

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def history(self, session_id: str) -> list[Turn]:
        """A copy of the turns, so callers can't change the stored history by accident."""
        return list(self._turns(session_id))

    def add(self, session_id: str, turn: Turn) -> None:
        turns = self._turns(session_id)
        turns.append(turn)
        del turns[: -self.max_turns]  # keep only the last max_turns

    def _turns(self, session_id: str) -> list[Turn]:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise UnknownSession(session_id) from None
