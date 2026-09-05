"""Provider-neutral task and recovery prompts."""

from __future__ import annotations

import json

from .controller import CompactRecovery, RetryRecovery, RuntimeRecovery


TASK_OBJECTIVE = "Complete the game with as few game actions as possible."
RECOVERY_TASK_OBJECTIVE = (
    "Continue the game with as few game actions as possible."
)


def compact_recovery_prompt(recovery: CompactRecovery) -> str:
    return _recovery_prompt(
        current_observation=recovery.current_observation,
        last_action_result=recovery.last_action_result,
        objective_history=recovery.objective_history,
        guide=recovery.guide,
        working_memory=recovery.working_memory,
    )


def retry_recovery_prompt(recovery: RetryRecovery) -> str:
    working_memory: object = recovery.working_memory
    if recovery.working_memory is not None:
        working_memory = {
            "saved_at": recovery.working_provenance,
            "content": recovery.working_memory,
        }
    return _recovery_prompt(
        current_observation=recovery.current_observation,
        last_action_result=recovery.last_action_result,
        objective_history=recovery.objective_history,
        guide=recovery.guide,
        working_memory=working_memory,
    )


def runtime_recovery_prompt(recovery: RuntimeRecovery) -> str:
    return "\n".join(
        [
            "The player runtime restarted without changing the game state.",
            "Environment record:",
            json.dumps(
                {
                    "current_observation": recovery.current_observation,
                    "last_action_result": recovery.last_action_result,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "",
            RECOVERY_TASK_OBJECTIVE,
        ]
    )


def native_compact_recovery_prompt(recovery: RuntimeRecovery) -> str:
    return "\n".join(
        [
            "Native context compaction completed without changing the game state.",
            "Current environment record:",
            json.dumps(
                {
                    "current_observation": recovery.current_observation,
                    "last_action_result": recovery.last_action_result,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "",
            RECOVERY_TASK_OBJECTIVE,
        ]
    )


def continuation_prompt(current_observation: dict[str, object]) -> str:
    return "\n".join(
        [
            "Current environment record:",
            json.dumps(
                current_observation,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "",
            RECOVERY_TASK_OBJECTIVE,
        ]
    )


def fresh_runtime_recovery_prompt(recovery: RuntimeRecovery) -> str:
    working_memory: object = recovery.working_memory
    if recovery.working_memory is not None:
        working_memory = {
            "saved_at": recovery.working_provenance,
            "content": recovery.working_memory,
        }
    return _recovery_prompt(
        current_observation=recovery.current_observation,
        last_action_result=recovery.last_action_result,
        objective_history=recovery.objective_history,
        guide=recovery.guide,
        working_memory=working_memory,
    )


def fresh_continuation_prompt(recovery: RuntimeRecovery) -> str:
    working_memory: object = recovery.working_memory
    if recovery.working_memory is not None:
        working_memory = {
            "saved_at": recovery.working_provenance,
            "content": recovery.working_memory,
        }
    return "\n".join(
        [
            "A fresh player is continuing from the unchanged game state.",
            _recovery_prompt(
                current_observation=recovery.current_observation,
                last_action_result=recovery.last_action_result,
                objective_history=recovery.objective_history,
                guide=recovery.guide,
                working_memory=working_memory,
            ),
        ]
    )


def _recovery_prompt(
    *,
    current_observation: dict[str, object],
    last_action_result: dict[str, object] | None,
    objective_history: dict[str, object],
    guide: object,
    working_memory: object,
) -> str:
    return "\n".join(
        [
            "Environment record:",
            json.dumps(
                {
                    "current_observation": current_observation,
                    "last_action_result": last_action_result,
                    "current_level_history": objective_history,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "",
            "Agent-authored notes:",
            json.dumps(
                {"guide": guide, "working_memory": working_memory},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "",
            RECOVERY_TASK_OBJECTIVE,
        ]
    )
