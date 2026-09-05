"""Claude subscription quota signals and credential leases."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence


DEFAULT_DRAIN_UTILIZATION = 0.90
DEFAULT_CAPACITY_POLL_SECONDS = 5.0


@dataclass(frozen=True)
class ClaudeRateLimitEvent:
    status: str
    rate_limit_type: str
    resets_at: int | None
    utilization: float | None = None
    surpassed_threshold: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "rate_limit_type": self.rate_limit_type,
            "resets_at": self.resets_at,
            "utilization": self.utilization,
            "surpassed_threshold": self.surpassed_threshold,
        }


def rate_limit_event_from_message(message: object) -> ClaudeRateLimitEvent | None:
    if not isinstance(message, dict) or message.get("type") != "rate_limit_event":
        return None
    info = message.get("rate_limit_info")
    if not isinstance(info, dict):
        return None
    status = info.get("status")
    rate_limit_type = info.get("rateLimitType")
    if not isinstance(status, str) or not isinstance(rate_limit_type, str):
        return None
    return ClaudeRateLimitEvent(
        status=status,
        rate_limit_type=rate_limit_type,
        resets_at=_optional_integer(info.get("resetsAt")),
        utilization=_optional_number(info.get("utilization")),
        surpassed_threshold=_optional_number(info.get("surpassedThreshold")),
    )


class ClaudeCredentialLease:
    """One running game holding one account slot."""

    def __init__(self, pool: ClaudeCredentialPool, account_slot: int) -> None:
        self._pool = pool
        self._account_slot = account_slot
        self._account_history = [account_slot]
        self._active = True

    @property
    def account_slot(self) -> int:
        return self._pool.account_slot(self)

    @property
    def oauth_token(self) -> str:
        return self._pool.oauth_token(self)

    @property
    def account_history(self) -> tuple[int, ...]:
        return self._pool.account_history(self)

    def observe_rate_limit(self, event: ClaudeRateLimitEvent) -> None:
        self._pool.observe_rate_limit(self, event)

    def should_yield(self) -> bool:
        return self._pool.should_yield(self)

    def relay(self) -> str:
        return self._pool.relay(self)

    def release(self) -> None:
        self._pool.release(self)


class ClaudeCredentialPool:
    """Coordinate account capacity and move games away from draining accounts."""

    def __init__(
        self,
        oauth_tokens: Sequence[str],
        worker_limits: Sequence[int],
        *,
        drain_utilization: float = DEFAULT_DRAIN_UTILIZATION,
        no_proactive_drain_accounts: Sequence[int] = (),
        clock: Callable[[], float] = time.time,
        capacity_poll_seconds: float = DEFAULT_CAPACITY_POLL_SECONDS,
    ) -> None:
        if not oauth_tokens or len(set(oauth_tokens)) != len(oauth_tokens):
            raise ValueError("Claude credentials must be non-empty and unique")
        if len(oauth_tokens) != len(worker_limits):
            raise ValueError("Claude worker limits must match the credential pool")
        if any(type(limit) is not int or limit < 0 for limit in worker_limits):
            raise ValueError("Claude worker limits must be non-negative integers")
        if not 0 < drain_utilization <= 1:
            raise ValueError("Claude drain utilization must be in (0, 1]")
        invalid_accounts = set(no_proactive_drain_accounts) - set(
            range(1, len(oauth_tokens) + 1)
        )
        if invalid_accounts:
            raise ValueError("Claude proactive-drain account index is invalid")
        if capacity_poll_seconds <= 0:
            raise ValueError("Claude capacity poll interval must be positive")

        self._tokens = {
            slot: token for slot, token in enumerate(oauth_tokens, start=1)
        }
        self._worker_limits = {
            slot: limit for slot, limit in enumerate(worker_limits, start=1)
        }
        self._active = {slot: 0 for slot in self._tokens}
        self._latest: dict[int, dict[str, ClaudeRateLimitEvent]] = {
            slot: {} for slot in self._tokens
        }
        self._blocked_until: dict[int, dict[str, float]] = {
            slot: {} for slot in self._tokens
        }
        self._drain_utilization = drain_utilization
        self._no_proactive_drain_accounts = frozenset(
            no_proactive_drain_accounts
        )
        self._clock = clock
        self._capacity_poll_seconds = capacity_poll_seconds
        self._waiting_relays = 0
        self._condition = threading.Condition(threading.RLock())

    def try_acquire(self, account_slot: int) -> ClaudeCredentialLease | None:
        with self._condition:
            self._validate_slot(account_slot)
            self._refresh_locked()
            if self._waiting_relays or not self._has_capacity_locked(account_slot):
                return None
            self._active[account_slot] += 1
            return ClaudeCredentialLease(self, account_slot)

    def set_worker_limits(self, worker_limits: Sequence[int]) -> None:
        if len(worker_limits) != len(self._tokens):
            raise ValueError("Claude worker limits must match the credential pool")
        if any(type(limit) is not int or limit < 0 for limit in worker_limits):
            raise ValueError("Claude worker limits must be non-negative integers")
        with self._condition:
            self._worker_limits = {
                slot: limit for slot, limit in enumerate(worker_limits, start=1)
            }
            self._condition.notify_all()

    def observe_account_rate_limit(
        self,
        account_slot: int,
        event: ClaudeRateLimitEvent,
    ) -> None:
        with self._condition:
            self._validate_slot(account_slot)
            self._record_rate_limit_locked(account_slot, event)

    def observe_rate_limit(
        self,
        lease: ClaudeCredentialLease,
        event: ClaudeRateLimitEvent,
    ) -> None:
        with self._condition:
            self._require_active_lease_locked(lease)
            self._record_rate_limit_locked(lease._account_slot, event)

    def should_yield(self, lease: ClaudeCredentialLease) -> bool:
        with self._condition:
            self._require_active_lease_locked(lease)
            self._refresh_locked()
            return bool(self._blocked_until[lease._account_slot])

    def relay(self, lease: ClaudeCredentialLease) -> str:
        with self._condition:
            self._require_active_lease_locked(lease)
            old_slot = lease._account_slot
            self._active[old_slot] -= 1
            lease._active = False
            self._waiting_relays += 1
            self._condition.notify_all()
            try:
                while True:
                    self._refresh_locked()
                    account_slot = self._best_available_slot_locked()
                    if account_slot is not None:
                        self._active[account_slot] += 1
                        lease._account_slot = account_slot
                        lease._account_history.append(account_slot)
                        lease._active = True
                        return self._tokens[account_slot]
                    self._condition.wait(timeout=self._next_poll_delay_locked())
            finally:
                self._waiting_relays -= 1
                self._condition.notify_all()

    def release(self, lease: ClaudeCredentialLease) -> None:
        with self._condition:
            if lease._pool is not self or not lease._active:
                return
            self._active[lease._account_slot] -= 1
            lease._active = False
            self._condition.notify_all()

    def account_slot(self, lease: ClaudeCredentialLease) -> int:
        with self._condition:
            self._require_lease_locked(lease)
            return lease._account_slot

    def oauth_token(self, lease: ClaudeCredentialLease) -> str:
        with self._condition:
            self._require_active_lease_locked(lease)
            return self._tokens[lease._account_slot]

    def account_history(self, lease: ClaudeCredentialLease) -> tuple[int, ...]:
        with self._condition:
            self._require_lease_locked(lease)
            return tuple(lease._account_history)

    def active_counts(self) -> dict[int, int]:
        with self._condition:
            return dict(self._active)

    def healthy_account_slots(self) -> tuple[int, ...]:
        with self._condition:
            self._refresh_locked()
            return tuple(
                slot for slot in self._tokens if not self._blocked_until[slot]
            )

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            self._refresh_locked()
            return {
                "drain_utilization": self._drain_utilization,
                "accounts": {
                    str(slot): {
                        "active": self._active[slot],
                        "worker_limit": self._worker_limits[slot],
                        "proactive_drain": (
                            slot not in self._no_proactive_drain_accounts
                        ),
                        "blocked_until": {
                            kind: reset_at if math.isfinite(reset_at) else None
                            for kind, reset_at in self._blocked_until[slot].items()
                        },
                        "latest": {
                            kind: event.as_dict()
                            for kind, event in self._latest[slot].items()
                        },
                    }
                    for slot in self._tokens
                },
            }

    def _record_rate_limit_locked(
        self,
        account_slot: int,
        event: ClaudeRateLimitEvent,
    ) -> None:
        self._refresh_locked()
        self._latest[account_slot][event.rate_limit_type] = event
        should_drain = event.status == "rejected" or (
            account_slot not in self._no_proactive_drain_accounts
            and event.utilization is not None
            and event.utilization >= self._drain_utilization
        )
        if should_drain:
            self._blocked_until[account_slot][event.rate_limit_type] = (
                float(event.resets_at)
                if event.resets_at is not None
                else math.inf
            )
        self._condition.notify_all()

    def _refresh_locked(self) -> None:
        now = self._clock()
        for events in self._latest.values():
            expired = [
                kind
                for kind, event in events.items()
                if event.resets_at is not None and event.resets_at <= now
            ]
            for kind in expired:
                del events[kind]
        for limits in self._blocked_until.values():
            expired = [kind for kind, reset_at in limits.items() if reset_at <= now]
            for kind in expired:
                del limits[kind]

    def _has_capacity_locked(self, account_slot: int) -> bool:
        return (
            not self._blocked_until[account_slot]
            and self._active[account_slot] < self._worker_limits[account_slot]
        )

    def _best_available_slot_locked(self) -> int | None:
        candidates = [
            slot for slot in self._tokens if self._has_capacity_locked(slot)
        ]
        if not candidates:
            return None

        def score(slot: int) -> tuple[float, float, int]:
            known_utilization = max(
                (
                    event.utilization
                    for event in self._latest[slot].values()
                    if event.utilization is not None
                ),
                default=0.0,
            )
            limit = self._worker_limits[slot]
            load = self._active[slot] / limit if limit else math.inf
            return known_utilization, load, slot

        return min(candidates, key=score)

    def _next_poll_delay_locked(self) -> float:
        now = self._clock()
        future_resets = [
            reset_at
            for limits in self._blocked_until.values()
            for reset_at in limits.values()
            if math.isfinite(reset_at) and reset_at > now
        ]
        if not future_resets:
            return self._capacity_poll_seconds
        return max(
            0.01,
            min(self._capacity_poll_seconds, min(future_resets) - now),
        )

    def _validate_slot(self, account_slot: int) -> None:
        if account_slot not in self._tokens:
            raise ValueError(f"Unknown Claude account slot {account_slot}")

    def _require_lease_locked(self, lease: ClaudeCredentialLease) -> None:
        if lease._pool is not self:
            raise ValueError("Claude credential lease belongs to another pool")

    def _require_active_lease_locked(self, lease: ClaudeCredentialLease) -> None:
        self._require_lease_locked(lease)
        if not lease._active:
            raise RuntimeError("Claude credential lease is not active")


def _optional_integer(value: object) -> int | None:
    return value if type(value) is int else None


def _optional_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
