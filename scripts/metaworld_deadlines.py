"""Absolute UTC and monotonic budget checks without any RL runtime dependencies."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_deadline(value: str) -> datetime:
    """Require an explicit timezone and normalize an ISO 8601 cutoff to UTC."""
    try:
        deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("deadline must be an ISO 8601 timestamp with a timezone") from error
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("deadline must include a timezone, for example 2026-11-09T20:00:00Z")
    return deadline.astimezone(timezone.utc)


class BudgetExpired(RuntimeError):
    """A global cutoff was observed; only output retention and cleanup may continue."""


class RunDeadline:
    """Latch the first global cutoff, even if the UTC clock subsequently moves back.

    Training reserves shorten both clocks' available allowance without expiring the
    whole run. Both global clocks are checked on every call, including reserved
    phases. Checks cannot interrupt a blocking environment/optimizer/I/O call;
    the next check stops scientific work, while cleanup may finish after cutoff.
    """

    def __init__(
        self,
        seconds: float,
        deadline_utc: datetime,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("run seconds must be positive and finite")
        if deadline_utc.tzinfo is None or deadline_utc.utcoffset() is None:
            raise ValueError("absolute deadline must include a timezone")
        self.seconds = seconds
        self.deadline_utc = deadline_utc.astimezone(timezone.utc)
        self.monotonic = monotonic
        self.now = now
        self.started_monotonic = monotonic()
        self.started_utc = now().astimezone(timezone.utc)
        self.cutoff_reason: str | None = None
        self.cutoff_phase: str | None = None
        self.cutoff_observed_utc: str | None = None
        self.cutoff_observed_elapsed: float | None = None

    def allowed(self, phase: str, *, reserve_seconds: float = 0) -> bool:
        if not math.isfinite(reserve_seconds) or not 0 <= reserve_seconds < self.seconds:
            raise ValueError("reserve must be nonnegative and smaller than the run budget")
        current_utc = self.now().astimezone(timezone.utc)
        elapsed = self.monotonic() - self.started_monotonic
        self._observe(phase, current_utc, elapsed)
        if self.cutoff_reason is not None:
            return False
        return (
            elapsed < self.seconds - reserve_seconds
            and (self.deadline_utc - current_utc).total_seconds() > reserve_seconds
        )

    def _observe(self, phase: str, current_utc: datetime, elapsed: float) -> None:
        if self.cutoff_reason is not None:
            return
        if current_utc >= self.deadline_utc:
            self.cutoff_reason = "absolute_utc_deadline"
        elif elapsed >= self.seconds:
            self.cutoff_reason = "monotonic_run_budget"
        if self.cutoff_reason is not None:
            self.cutoff_phase = phase
            self.cutoff_observed_utc = current_utc.isoformat()
            self.cutoff_observed_elapsed = elapsed

    def require(self, phase: str) -> None:
        if not self.allowed(phase):
            raise BudgetExpired(f"{self.cutoff_reason} at {phase}")

    def record(self, manifest: dict[str, Any]) -> None:
        """Record actual finish clocks, including cutoff during output/cleanup."""
        finished = self.now().astimezone(timezone.utc)
        elapsed = self.monotonic() - self.started_monotonic
        self._observe("finalize", finished, elapsed)
        manifest.update(
            started_at_utc=self.started_utc.isoformat(),
            finished_at_utc=finished.isoformat(),
            elapsed_seconds=elapsed,
            utc_elapsed_seconds=(finished - self.started_utc).total_seconds(),
            absolute_deadline_utc=self.deadline_utc.isoformat(),
            budget_compliant=self.cutoff_reason is None,
            cutoff_reason=self.cutoff_reason,
            cutoff_phase=self.cutoff_phase,
            cutoff_observed_at_utc=self.cutoff_observed_utc,
            cutoff_observed_monotonic_elapsed_seconds=self.cutoff_observed_elapsed,
        )
        if self.cutoff_reason is not None:
            if manifest.get("status") != "ERROR":
                manifest["status"] = "STOPPED_AT_BUDGET"
            manifest["validation_complete"] = False


class TrainingDeadline:
    """Dependency-free training callback state shared with the SB3 adapter."""

    def __init__(self, budget: RunDeadline, *, reserve_seconds: float = 0) -> None:
        self.budget = budget
        self.reserve_seconds = reserve_seconds
        self.stopped_for_budget = False
        self.cutoff_reason: str | None = None

    def on_step(self) -> bool:
        if not self.budget.allowed("training_callback", reserve_seconds=self.reserve_seconds):
            self.stopped_for_budget = True
            self.cutoff_reason = self.budget.cutoff_reason or "training_evaluation_reserve"
            return False
        return True


def _guard_method(target: Any, name: str, budget: RunDeadline, phase: str) -> None:
    """Keep the object's runtime type while checking entry to one bound method."""
    original = getattr(target, name)

    def checked(*args: Any, **kwargs: Any) -> Any:
        budget.require(phase)
        return original(*args, **kwargs)

    setattr(target, name, checked)


def guard_env(env: Any, budget: RunDeadline) -> Any:
    """Check reset/step at their boundary, including calls made inside SB3."""
    _guard_method(env, "reset", budget, "environment_reset")
    _guard_method(env, "step", budget, "environment_step")
    return env


@contextmanager
def guard_model(model: Any, budget: RunDeadline) -> Iterator[None]:
    """Guard optimizer entry, then restore the model before checkpoint serialization."""
    missing = object()
    previous = vars(model).get("train", missing)
    _guard_method(model, "train", budget, "optimizer_update")
    try:
        yield
    finally:
        if previous is missing:
            del model.train
        else:
            model.train = previous
