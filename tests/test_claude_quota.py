from vista_arc3.claude.quota import (
    ClaudeCredentialPool,
    ClaudeRateLimitEvent,
    rate_limit_event_from_message,
)


def test_rate_limit_event_parses_native_stream_message() -> None:
    event = rate_limit_event_from_message(
        {
            "type": "rate_limit_event",
            "rate_limit_info": {
                "status": "allowed_warning",
                "rateLimitType": "five_hour",
                "resetsAt": 1234,
                "utilization": 0.9,
                "surpassedThreshold": 0.9,
            },
        }
    )

    assert event == ClaudeRateLimitEvent(
        status="allowed_warning",
        rate_limit_type="five_hour",
        resets_at=1234,
        utilization=0.9,
        surpassed_threshold=0.9,
    )
    assert rate_limit_event_from_message({"type": "result"}) is None


def test_warning_drains_account_and_relay_moves_the_live_game() -> None:
    now = [1000.0]
    pool = ClaudeCredentialPool(
        ("account-one", "account-two"),
        (1, 1),
        clock=lambda: now[0],
    )
    draining = pool.try_acquire(1)
    healthy = pool.try_acquire(2)
    assert draining is not None
    assert healthy is not None

    draining.observe_rate_limit(
        ClaudeRateLimitEvent(
            status="allowed_warning",
            rate_limit_type="five_hour",
            resets_at=1200,
            utilization=0.90,
            surpassed_threshold=0.90,
        )
    )
    assert draining.should_yield() is True
    assert pool.try_acquire(1) is None

    healthy.release()
    assert draining.relay() == "account-two"
    assert draining.account_history == (1, 2)
    assert pool.active_counts() == {1: 0, 2: 1}
    draining.release()

    now[0] = 1201
    replacement = pool.try_acquire(1)
    assert replacement is not None
    replacement.release()


def test_warning_from_one_game_drains_every_lease_on_the_account() -> None:
    pool = ClaudeCredentialPool(
        ("account-one", "account-two"),
        (2, 1),
        clock=lambda: 1000,
    )
    first = pool.try_acquire(1)
    second = pool.try_acquire(1)
    assert first is not None
    assert second is not None

    first.observe_rate_limit(
        ClaudeRateLimitEvent(
            status="allowed_warning",
            rate_limit_type="five_hour",
            resets_at=2000,
            utilization=0.90,
        )
    )

    assert first.should_yield() is True
    assert second.should_yield() is True
    assert pool.healthy_account_slots() == (2,)
    first.release()
    second.release()


def test_weekly_warning_below_drain_threshold_stays_available() -> None:
    pool = ClaudeCredentialPool(("account",), (1,), clock=lambda: 1000)
    pool.observe_account_rate_limit(
        1,
        ClaudeRateLimitEvent(
            status="allowed_warning",
            rate_limit_type="seven_day",
            resets_at=2000,
            utilization=0.50,
            surpassed_threshold=0.50,
        ),
    )

    lease = pool.try_acquire(1)
    assert lease is not None
    assert lease.should_yield() is False
    lease.release()


def test_rejected_account_is_unavailable_even_without_utilization() -> None:
    pool = ClaudeCredentialPool(("account",), (1,), clock=lambda: 1000)
    pool.observe_account_rate_limit(
        1,
        ClaudeRateLimitEvent(
            status="rejected",
            rate_limit_type="five_hour",
            resets_at=2000,
        ),
    )

    assert pool.try_acquire(1) is None


def test_account_can_ignore_proactive_drain_but_still_honor_rejection() -> None:
    pool = ClaudeCredentialPool(
        ("account-one", "account-two"),
        (1, 1),
        no_proactive_drain_accounts=(1,),
        clock=lambda: 1000,
    )
    lease = pool.try_acquire(1)
    assert lease is not None

    lease.observe_rate_limit(
        ClaudeRateLimitEvent(
            status="allowed_warning",
            rate_limit_type="seven_day",
            resets_at=2000,
            utilization=0.99,
        )
    )
    assert lease.should_yield() is False
    assert pool.snapshot()["accounts"]["1"]["proactive_drain"] is False

    lease.observe_rate_limit(
        ClaudeRateLimitEvent(
            status="rejected",
            rate_limit_type="seven_day",
            resets_at=2000,
        )
    )
    assert lease.should_yield() is True
    lease.release()


def test_zero_worker_limit_retires_account_after_active_lease_finishes() -> None:
    pool = ClaudeCredentialPool(
        ("account-one", "account-two"),
        (1, 1),
        clock=lambda: 1000,
    )
    lease = pool.try_acquire(1)
    assert lease is not None

    pool.set_worker_limits((0, 1))
    assert lease.oauth_token == "account-one"
    assert pool.try_acquire(1) is None

    lease.release()
    assert pool.try_acquire(1) is None
    replacement = pool.try_acquire(2)
    assert replacement is not None
    replacement.release()


def test_expired_usage_window_removes_stale_utilization() -> None:
    now = [1000.0]
    pool = ClaudeCredentialPool(
        ("account-one", "account-two"),
        (1, 1),
        clock=lambda: now[0],
    )
    pool.observe_account_rate_limit(
        1,
        ClaudeRateLimitEvent(
            status="allowed_warning",
            rate_limit_type="seven_day",
            resets_at=1100,
            utilization=0.89,
        ),
    )

    now[0] = 1101
    lease = pool.try_acquire(1)
    assert lease is not None
    assert pool.snapshot()["accounts"]["1"]["latest"] == {}
    lease.release()


def test_unknown_reset_is_serialized_without_nonstandard_infinity() -> None:
    pool = ClaudeCredentialPool(("account",), (1,))
    pool.observe_account_rate_limit(
        1,
        ClaudeRateLimitEvent(
            status="rejected",
            rate_limit_type="five_hour",
            resets_at=None,
        ),
    )

    assert pool.snapshot()["accounts"]["1"]["blocked_until"] == {
        "five_hour": None
    }
