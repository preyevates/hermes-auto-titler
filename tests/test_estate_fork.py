"""Estate fork guarantees: startup_requeue off switch and title_cutoff."""
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
