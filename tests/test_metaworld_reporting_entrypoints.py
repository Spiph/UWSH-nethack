"""Source CLI entrypoints require reporter readiness before a fake scientific body."""

from __future__ import annotations

import importlib
import json
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import metaworld_development as development
from scripts import metaworld_reporting_sidecar as sidecar
from scripts import metaworld_source_continuation as continuation
from scripts import metaworld_technical_retry as retry


@pytest.mark.parametrize("kind", ["development", "retry", "continuation"])
def test_actual_source_cli_wraps_run_in_ready_observer_without_native_work(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    active = False

    @contextmanager
    def observe(path: Path, artifact_limit: int) -> Any:
        nonlocal active
        assert path.name == "development_manifest.json"
        assert path.parent.is_dir() and artifact_limit == 8 * 1024**2
        active = True
        events.append("observer_ready")
        try:
            yield {}
        finally:
            active = False
            events.append("observer_final")

    class FakeRunner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            root = tmp_path / "new-fake-artifacts"
            root.mkdir()
            self.store = SimpleNamespace(root=root, limit=8 * 1024**2)

        def run(self) -> dict[str, Any]:
            assert active
            events.append("science")
            return {"status": "COMPLETE_PASS", "ledger": {}}

    config, approval, binding = (tmp_path / name for name in ("config", "approval", "binding"))
    saved_config = {"deadline_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}
    config.write_text(json.dumps(saved_config))
    for path in (approval, binding):
        path.write_text(json.dumps({}))
    monkeypatch.setattr(sidecar, "reporting_sidecar", observe)
    load = SimpleNamespace(load=lambda path: SimpleNamespace())
    imported = importlib.import_module

    def import_plan(name: str) -> Any:
        if name == "scripts.metaworld_development_plan":
            return SimpleNamespace(DevelopmentPlan=load)
        return imported(name)

    monkeypatch.setattr(importlib, "import_module", import_plan)
    if kind == "development":
        monkeypatch.setattr(development, "DevelopmentRunner", FakeRunner)
        code = development.main(["--config", str(config), "--approval", str(approval), "--run"])
    elif kind == "retry":
        monkeypatch.setattr(retry.TechnicalRetryContract, "load", lambda path: object())
        monkeypatch.setattr(retry, "validate_retry_approval", lambda *args: None)
        monkeypatch.setattr(retry, "prior_evidence_lock", lambda *args: nullcontext())
        monkeypatch.setattr(retry, "TechnicalRetryRunner", FakeRunner)
        code = retry.main(
            [
                "--config",
                str(config),
                "--contract",
                str(config),
                "--approval",
                str(approval),
                "--run",
            ]
        )
    else:
        monkeypatch.setattr(continuation, "extension_configuration", lambda: saved_config)
        monkeypatch.setattr(continuation, "validate_extension_approval", lambda *args: None)
        monkeypatch.setattr(
            continuation,
            "ExtensionPlan",
            lambda *args: SimpleNamespace(
                deadline_utc=datetime.now(timezone.utc) + timedelta(hours=1)
            ),
        )
        monkeypatch.setattr(
            continuation, "inherited_monotonic_cutoff", lambda *args: time.monotonic() + 3600
        )
        monkeypatch.setattr(continuation, "ContinuationRunner", FakeRunner)
        code = continuation.main(
            [
                "--config",
                str(config),
                "--approval",
                str(approval),
                "--binding",
                str(binding),
                "--monotonic-cutoff",
                "1",
            ]
        )
    assert code == 0
    assert events == ["observer_ready", "science", "observer_final"]
