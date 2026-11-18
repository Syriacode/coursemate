import re

import pytest

from coursemate.chat import Turn
from coursemate.sessions import SessionStore, UnknownSession


def turn(n: int) -> Turn:
    return Turn(f"question {n}", f"answer {n}")


def test_create_gives_different_random_ids():
    store = SessionStore()
    first, second = store.create(), store.create()

    assert first != second
    assert re.fullmatch(r"[0-9a-f]{32}", first)
    assert store.exists(first) and store.exists(second)
    assert not store.exists("nope")


def test_a_new_session_is_empty_and_add_keeps_the_order():
    store = SessionStore()
    session = store.create()
    assert store.history(session) == []

    store.add(session, turn(1))
    store.add(session, turn(2))

    assert store.history(session) == [turn(1), turn(2)]


def test_history_returns_a_copy():
    store = SessionStore()
    session = store.create()
    store.add(session, turn(1))

    store.history(session).append(turn(99))

    assert store.history(session) == [turn(1)]


def test_only_the_last_max_turns_are_kept():
    store = SessionStore(max_turns=2)
    session = store.create()
    for n in (1, 2, 3):
        store.add(session, turn(n))

    assert store.history(session) == [turn(2), turn(3)]


def test_default_keeps_five_turns():
    store = SessionStore()
    session = store.create()
    for n in range(1, 8):
        store.add(session, turn(n))

    assert store.history(session) == [turn(n) for n in range(3, 8)]


def test_unknown_session_raises_a_key_error_subclass():
    store = SessionStore()

    with pytest.raises(UnknownSession):
        store.history("nope")
    with pytest.raises(UnknownSession):
        store.add("nope", turn(1))
    assert issubclass(UnknownSession, KeyError)


def test_when_full_the_oldest_session_is_dropped():
    store = SessionStore(max_sessions=2)
    first, second, third = store.create(), store.create(), store.create()

    assert not store.exists(first)
    assert store.exists(second) and store.exists(third)
    with pytest.raises(UnknownSession):
        store.history(first)
