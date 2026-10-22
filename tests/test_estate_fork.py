"""Estate fork guarantees: startup_requeue off switch, title_cutoff and task_context."""
import datetime as dt
from types import SimpleNamespace

import pytest

from hermes_auto_titler.config import DEFAULTS, coerce_value, parse_cutoff
from hermes_auto_titler.titler import AutoTitler
from test_titler import MSGS, FakeDB, FakeLlm


class DatedDB(FakeDB):
    def __init__(self, started_at, **kw):
        super().__init__(**kw)
        self.started_at = started_at

    def get_session(self, sid):
        return None if self.started_at is None else {"id": sid, "started_at": self.started_at}


def titler(db, **cfg):
    llm = FakeLlm('{"action":"rename","title":"Automatic Updater: nightly run fix"}')
    return AutoTitler(SimpleNamespace(llm=llm), {**DEFAULTS, "rename_confirmations": 0, **cfg}, db=db), llm


def test_defaults_keep_upstream_behaviour():
    assert DEFAULTS["startup_requeue"] is True and DEFAULTS["title_cutoff"] == ""


def test_parse_cutoff_forms():
    assert parse_cutoff("") is None and parse_cutoff(None) is None and parse_cutoff("nonsense") is None
    assert parse_cutoff("2026-10-22T18:00:00+00:00") == dt.datetime(2026, 10, 22, 18, tzinfo=dt.timezone.utc).timestamp()
    assert parse_cutoff("1792000000") == 1792000000.0
    assert parse_cutoff(dt.date(2026, 10, 22)) is not None
    with pytest.raises(ValueError):
        coerce_value("title_cutoff", "not-a-date")


def test_session_before_cutoff_is_never_auto_titled():
    db = DatedDB(1000.0, messages=MSGS, title=None, source=None)
    t, llm = titler(db, title_cutoff="2000")
    t._submit_eval("old")
    t._requeue_untitled_sessions()
    assert llm.calls == [] and db.calls == [] and db.title is None


def test_session_without_start_time_is_treated_as_past():
    db = DatedDB(None, messages=MSGS, title=None, source=None)
    t, llm = titler(db, title_cutoff="2000")
    assert t._before_cutoff("s1") is True


def test_session_after_cutoff_is_titled():
    db = DatedDB(5000.0, messages=MSGS, title=None, source=None)
    t, _ = titler(db, title_cutoff="2000")
    assert t._before_cutoff("new") is False


def test_startup_requeue_false_skips_sweep(monkeypatch):
    db = DatedDB(5000.0, messages=MSGS, title=None, source=None)
    t, _ = titler(db, startup_requeue=False)
    swept = []
    monkeypatch.setattr(t, "_requeue_untitled_sessions", lambda: swept.append(1))
    monkeypatch.setattr("threading.Thread.start", lambda self: self.run())
    monkeypatch.setattr(t, "_retry_failed_sessions", lambda: (_ for _ in ()).throw(SystemExit), raising=False)
    try:
        t.start_retry_loop()
    except SystemExit:
        pass
    assert swept == []


def test_user_title_never_overwritten_after_cutoff():
    db = DatedDB(5000.0, messages=MSGS, title="My own title", source="user")
    t, _ = titler(db, title_cutoff="2000")
    t.evaluate("s1", force=True)
    assert (db.title, db.source) == ("My own title", "user")


# --- task_context (estate fork, 0.3.0+estate.2) -------------------------------------------

CARD_CONV = [
    {"role": "user", "content": "work kanban task t_d44358a1"},
    {"role": "assistant", "content": ""},
    {"role": "tool", "content": '{"task": {"id": "t_d44358a1", "title": "Renovate: nightly run fails on lookup", "body": "x"}}'},
    {"role": "assistant", "content": "Done. The lookup now passes."},
]


class ConvDB(FakeDB):
    def get_messages_as_conversation(self, session_id, include_ancestors=False, include_compacted=False):
        return self.messages


def _ctx(conv, **kw):
    from hermes_auto_titler.messages import load_context_with_summary
    return load_context_with_summary(ConvDB(messages=conv), "s1", recent_turns=2, include_all_user=True,
                                     opening_turns=2, **kw)


def test_task_context_off_by_default_keeps_upstream_context():
    assert DEFAULTS["task_context"] is False
    _, all_user, _, _ = _ctx(CARD_CONV)
    assert all_user and all("[card:" not in t for _, t in all_user)


def test_task_context_adds_card_title_to_starting_user_turn():
    recent, all_user, _, _ = _ctx(CARD_CONV, task_context=True)
    assert any("[card: Renovate: nightly run fails on lookup]" in t for r, t in recent if r == "user")
    assert any("[card: Renovate: nightly run fails on lookup]" in t for _, t in all_user)
    assert all(r != "tool" for r, _ in recent)


def test_task_card_title_parses_truncated_and_ignores_other_tools():
    from hermes_auto_titler.messages import task_card_title
    assert task_card_title('{"task": {"id": "t_1", "title": "SEC: two \\"quoted\\" words", "body": "trunc') == 'SEC: two "quoted" words'
    assert task_card_title('{"output": "ls", "exit_code": 0}') is None
    assert task_card_title("plain text") is None
    assert task_card_title({"task": {"title": "  a   b  "}}) == "a b"


def test_task_context_only_first_card_is_used():
    conv = CARD_CONV + [{"role": "user", "content": "now look at another"},
                        {"role": "tool", "content": '{"task": {"id": "t_2", "title": "Other card"}}'}]
    _, all_user, _, _ = _ctx(conv, task_context=True)
    assert sum("[card:" in t for _, t in all_user) == 1 and not any("Other card" in t for _, t in all_user)


def test_titler_passes_task_context_to_prompt():
    db = DatedDB(5000.0, messages=CARD_CONV, title=None, source=None)
    t, llm = titler(db, task_context=True)
    t.evaluate("s1", force=True)
    sent = " ".join(str(m.get("content")) for call in llm.calls for m in call.get("messages", []))
    assert "Renovate: nightly run fails on lookup" in sent
