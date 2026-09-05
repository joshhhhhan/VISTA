import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from arcengine import GameAction, GameState
from PIL import Image

import vista_arc3.codex.controller as controller_module
from vista_arc3.codex.controller import (
    GameController,
    validate_tool_action,
)
from vista_arc3.codex.dispatcher import GameToolDispatcher, execution_from_response


class FakeEnvironment:
    def __init__(self, observations, actions=None):
        self.observations = list(observations)
        self.action_space = actions or [GameAction.ACTION1, GameAction.ACTION6]
        self.calls = []

    def step(self, action, data=None, reasoning=None):
        self.calls.append(
            {
                "method": "step",
                "action": action,
                "data": data,
                "reasoning": reasoning,
            }
        )
        return self.observations.pop(0)

    def reset(self):
        self.calls.append(
            {
                "method": "reset",
                "action": GameAction.RESET,
                "data": None,
                "reasoning": None,
            }
        )
        return self.observations.pop(0)


def observation(
    value: int = 0,
    *,
    state: GameState = GameState.NOT_FINISHED,
    frames: int = 1,
    levels_completed: int = 0,
    win_levels: int = 1,
):
    return SimpleNamespace(
        game_id="private-game-id",
        guid="private-guid",
        state=state,
        levels_completed=levels_completed,
        win_levels=win_levels,
        available_actions=[1, 6],
        action_input=None,
        full_reset=False,
        frame=[np.full((64, 64), value + index, dtype=np.uint8) for index in range(frames)],
    )


def make_controller(
    tmp_path: Path,
    *,
    next_observations=None,
    max_steps: int = 10,
    max_invalid_retries: int = 2,
    observation_mode: str = "vision",
    initial_observation=None,
    compact_restore_marker: Path | None = None,
    compact_guide_snapshot: Path | None = None,
    compact_checkpoint_marker: Path | None = None,
    compact_checkpoint_ready: Path | None = None,
    retry_boundary_marker: Path | None = None,
    reset_starts_fresh_session: bool = True,
    working_path: Path | None = None,
    guide_path: Path | None = None,
    render_scale: int = 16,
    render_grid: bool = False,
):
    private = tmp_path / "private"
    frames = tmp_path / "visible" / "screenshots"
    private.mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)
    if working_path is None:
        working_path = tmp_path / "visible" / "WORKING.md"
    env = FakeEnvironment(next_observations or [observation(1)])
    controller = GameController(
        env=env,
        initial_observation=initial_observation or observation(frames=2),
        private_dir=private,
        frames_dir=frames,
        max_steps=max_steps,
        max_invalid_retries=max_invalid_retries,
        observation_mode=observation_mode,
        render_scale=render_scale,
        render_grid=render_grid,
        compact_restore_marker=compact_restore_marker,
        compact_guide_snapshot=compact_guide_snapshot,
        compact_checkpoint_marker=compact_checkpoint_marker,
        compact_checkpoint_ready=compact_checkpoint_ready,
        retry_boundary_marker=retry_boundary_marker,
        reset_starts_fresh_session=reset_starts_fresh_session,
        working_path=working_path,
        guide_path=guide_path,
    )
    return controller, env, private, frames


def test_click_coordinates_are_visual_pixels() -> None:
    validation = validate_tool_action(
        {"action": "ACTION6", "x": 37 * 16 + 8, "y": 12 * 16 + 8},
        [GameAction.ACTION6],
    )

    assert validation.ok is True
    assert validation.data == {"x": 37, "y": 12}
    assert validation.parsed == {
        "action": "ACTION6",
        "x": 600,
        "y": 200,
        "environment_x": 37,
        "environment_y": 12,
    }


def test_reset_requires_a_bounded_retry_state() -> None:
    missing = validate_tool_action(
        {"action": "RESET"},
        [GameAction.RESET],
    )
    unrelated = validate_tool_action(
        {"action": "ACTION1", "retry_state": "Not applicable."},
        [GameAction.ACTION1],
    )
    oversized = validate_tool_action(
        {"action": "RESET", "retry_state": "x" * (16 * 1024 + 1)},
        [GameAction.RESET],
    )
    valid = validate_tool_action(
        {
            "action": "RESET",
            "retry_state": "  Preserve the failed attempt's useful result.  ",
        },
        [GameAction.RESET],
    )

    assert missing.ok is False
    assert missing.error is not None and "requires retry_state" in missing.error
    assert unrelated.ok is False
    assert unrelated.error == "retry_state is only used with RESET."
    assert oversized.ok is False
    assert valid.ok is True
    assert valid.retry_state == "Preserve the failed attempt's useful result."
    assert valid.parsed == {"action": "RESET", "x": None, "y": None}
    assert "retry_state" not in valid.parsed


def test_same_session_reset_needs_no_retry_state() -> None:
    valid = validate_tool_action(
        {"action": "RESET"},
        [GameAction.RESET],
        reset_requires_retry_state=False,
    )
    redundant = validate_tool_action(
        {"action": "RESET", "retry_state": "Redundant continuation."},
        [GameAction.RESET],
        reset_requires_retry_state=False,
    )

    assert valid.ok is True
    assert valid.retry_state is None
    assert valid.parsed == {"action": "RESET", "x": None, "y": None}
    assert redundant.ok is False
    assert redundant.error == "Unexpected tool argument."


def test_512_render_and_click_coordinates_share_one_display_size(tmp_path: Path) -> None:
    controller, env, _, frames = make_controller(
        tmp_path,
        render_scale=8,
        render_grid=True,
    )

    response = controller.handle(
        {
            "method": "play",
            "request_id": 1,
            "arguments": {"action": "ACTION6", "x": 37 * 8 + 4, "y": 12 * 8 + 4},
        }
    )

    assert response["ok"] is True
    assert env.calls[0]["data"] == {"x": 37, "y": 12}
    with Image.open(sorted(frames.glob("turn_001_frame_*.png"))[-1]) as image:
        assert image.size == (512, 512)
        assert image.getpixel((0, 0)) == tuple(
            (channel * 3) // 4 for channel in image.getpixel((1, 1))
        )
        assert image.getpixel((8, 4)) == tuple(
            (channel * 3) // 4 for channel in image.getpixel((9, 4))
        )
    invalid = validate_tool_action(
        {"action": "ACTION6", "x": 512, "y": 0},
        [GameAction.ACTION6],
        display_size=512,
    )
    assert invalid.ok is False
    assert invalid.error == "ACTION6 coordinates must be in 0..511."


def test_unexpected_play_argument_does_not_execute_action(tmp_path: Path) -> None:
    controller, env, _, _ = make_controller(tmp_path, render_scale=8)

    response = controller.handle(
        {
            "method": "play",
            "request_id": 1,
            "arguments": {"action": "ACTION1", "unexpected": True},
        }
    )

    assert response["ok"] is False
    assert response["metadata"]["error"] == "Unexpected tool argument."
    assert env.calls == []


def test_invalid_action_does_not_advance_or_repeat_image(tmp_path: Path) -> None:
    controller, env, _, _ = make_controller(tmp_path)

    response = controller.handle(
        {"method": "play", "request_id": 1, "arguments": {"action": "ACTION3"}}
    )

    assert response["ok"] is False
    assert response["metadata"]["action_applied"] is False
    assert "image" not in response
    assert env.calls == []
    assert controller.step_index == 0


def test_valid_action_archives_all_frames_but_returns_last(tmp_path: Path) -> None:
    controller, env, private, frames = make_controller(
        tmp_path,
        next_observations=[observation(3, frames=3)],
    )

    response = controller.handle(
        {
            "method": "play",
            "request_id": 2,
            "arguments": {"action": "ACTION1"},
        }
    )

    assert response["ok"] is True
    assert response["metadata"]["turn"] == 1
    assert response["metadata"]["visual"] == {
        "turn": 1,
        "frame_count": 3,
        "final_frame": 2,
        "width": 1024,
        "height": 1024,
    }
    assert "remaining_actions" not in response["metadata"]
    assert "latest_visual" not in response["metadata"]
    assert len(list(frames.glob("turn_001_frame_*.png"))) == 3
    assert base64.b64decode(response["image"]["data"]).startswith(b"\x89PNG")
    assert env.calls[0]["action"] == GameAction.ACTION1
    assert env.calls[0]["reasoning"] is None
    log = json.loads((private / "action_log.jsonl").read_text().splitlines()[0])
    assert log["validation"]["parsed"]["action"] == "ACTION1"
    assert "batch_index" not in log
    assert "batch_size" not in log


def test_inspect_returns_one_final_visual_without_changing_state(
    tmp_path: Path,
) -> None:
    controller, env, _, _ = make_controller(tmp_path, render_scale=8)

    response = controller.handle(
        {
            "method": "inspect",
            "request_id": 1,
            "arguments": {
                "question": "What is visible now?",
                "views": [{"label": "current overview", "turn": 0}],
            },
        }
    )

    assert response["ok"] is True
    assert response["metadata"] == {
        "visual_kind": "inspection",
        "current_state_unchanged": True,
        "view_count": 1,
        "views": [
            {
                "index": 1,
                "turn": 0,
                "frame": 1,
                "available_frames": 2,
                "available_frame_range": [0, 1],
                "region": {"x": 0, "y": 0, "width": 512, "height": 512},
                "image_size": {"width": 512, "height": 512},
            }
        ],
    }
    assert len(response["images"]) == 1
    with Image.open(
        io.BytesIO(base64.b64decode(response["images"][0]["data"]))
    ) as image:
        assert image.size == (512, 512)
    assert env.calls == []
    assert controller.step_index == 0


def test_inspect_renders_independent_views_in_request_order(tmp_path: Path) -> None:
    frame = np.zeros((64, 64), dtype=np.uint8)
    frame[10:13, 20:22] = np.array([[1, 2], [3, 4], [5, 6]])
    initial = observation()
    initial.frame = [frame]
    controller, env, _, frames = make_controller(
        tmp_path,
        render_scale=8,
        render_grid=True,
        initial_observation=initial,
    )

    response = controller.handle(
        {
            "method": "inspect",
            "request_id": 1,
            "arguments": {
                "question": "How do these regions differ?",
                "views": [
                    {
                        "label": "changed region",
                        "turn": 0,
                        "frame": 0,
                        "region": {"x": 161, "y": 81, "width": 14, "height": 22},
                    },
                    {
                        "label": "control region",
                        "turn": 0,
                        "frame": 0,
                        "region": {"x": 0, "y": 0, "width": 8, "height": 8},
                    },
                ]
            },
        }
    )

    assert response["ok"] is True
    assert response["metadata"]["view_count"] == 2
    assert response["metadata"]["views"][0]["region"] == {
        "x": 161,
        "y": 81,
        "width": 14,
        "height": 22,
    }
    assert response["metadata"]["views"][0]["image_size"] == {
        "width": 325,
        "height": 512,
    }
    assert response["metadata"]["views"][1]["index"] == 2
    assert len(response["images"]) == 2
    with Image.open(
        io.BytesIO(base64.b64decode(response["images"][0]["data"]))
    ) as image:
        actual = np.asarray(image)
    with Image.open(
        frames / "turn_000_frame_00.png"
    ) as archived:
        expected = archived.convert("RGB").crop((161, 81, 175, 103)).resize(
            (325, 512),
            Image.Resampling.NEAREST,
        )
    assert np.array_equal(actual, np.asarray(expected))
    with Image.open(
        io.BytesIO(base64.b64decode(response["images"][1]["data"]))
    ) as image:
        assert image.size == (512, 512)
    assert env.calls == []
    assert controller.step_index == 0


def test_inspect_is_unavailable_without_visual_observations(tmp_path: Path) -> None:
    controller, env, _, _ = make_controller(tmp_path, observation_mode="grid")

    response = controller.handle(
        {
            "method": "inspect",
            "arguments": {
                "question": "What is visible?",
                "views": [{"label": "overview", "turn": 0}],
            },
        }
    )

    assert response == {
        "ok": False,
        "metadata": {"error": "Visual observations are unavailable."},
    }
    assert env.calls == []


def test_read_pixels_keeps_rgb_symbols_stable_across_calls(
    tmp_path: Path,
) -> None:
    controller, env, _, frames = make_controller(tmp_path, render_scale=8)
    colors = [
        (10, 20, 30),
        (40, 50, 60),
        (70, 80, 90),
        (100, 110, 120),
    ]
    supplied = Image.new("RGB", (512, 512), colors[3])
    supplied.paste(colors[0], (256, 0, 512, 256))
    supplied.paste(colors[2], (0, 256, 256, 512))
    supplied.paste(colors[1], (256, 256, 512, 512))
    supplied.save(frames / "turn_000_frame_00.png")
    later = supplied.copy()
    later.paste((1, 2, 3), (0, 0, 256, 256))
    later.save(frames / "turn_000_frame_01.png")

    response = controller.handle(
        {
            "method": "read_pixels",
            "request_id": 1,
            "arguments": {
                "question": "Are the four quadrants distinct?",
                "views": [
                    {
                        "label": "all quadrants",
                        "turn": 0,
                        "frame": 0,
                        "region": {
                            "x": 0,
                            "y": 0,
                            "width": 512,
                            "height": 512,
                        },
                        "rows": 2,
                        "columns": 2,
                    },
                    {
                        "label": "right quadrants",
                        "turn": 0,
                        "frame": 0,
                        "region": {
                            "x": 256,
                            "y": 0,
                            "width": 256,
                            "height": 512,
                        },
                        "rows": 2,
                        "columns": 1,
                    },
                ],
            },
        }
    )

    assert response == {
        "ok": True,
        "metadata": {
            "visual_kind": "pixel_readout",
            "current_state_unchanged": True,
            "sampling": "equal-bin-centers",
            "palette": {
                "0": "#0A141E",
                "1": "#28323C",
                "2": "#46505A",
                "3": "#646E78",
            },
            "sample_count": 6,
            "view_count": 2,
            "views": [
                {
                    "index": 1,
                    "turn": 0,
                    "frame": 0,
                    "available_frames": 2,
                    "available_frame_range": [0, 1],
                    "region": {
                        "x": 0,
                        "y": 0,
                        "width": 512,
                        "height": 512,
                    },
                    "rows": 2,
                    "columns": 2,
                    "samples": ["30", "21"],
                },
                {
                    "index": 2,
                    "turn": 0,
                    "frame": 0,
                    "available_frames": 2,
                    "available_frame_range": [0, 1],
                    "region": {
                        "x": 256,
                        "y": 0,
                        "width": 256,
                        "height": 512,
                    },
                    "rows": 2,
                    "columns": 1,
                    "samples": ["0", "1"],
                },
            ],
        },
    }
    serialized = json.dumps(response)
    assert "Are the four quadrants distinct?" not in serialized
    assert "all quadrants" not in serialized
    assert "right quadrants" not in serialized
    assert env.calls == []
    assert controller.step_index == 0

    later_response = controller.handle(
        {
            "method": "read_pixels",
            "arguments": {
                "question": "Did an earlier color keep its symbol?",
                "views": [
                    {
                        "label": "new lower RGB",
                        "turn": 0,
                        "frame": 1,
                        "region": {
                            "x": 0,
                            "y": 0,
                            "width": 256,
                            "height": 256,
                        },
                        "rows": 1,
                        "columns": 1,
                    },
                    {
                        "label": "previously assigned RGB",
                        "turn": 0,
                        "frame": 1,
                        "region": {
                            "x": 0,
                            "y": 256,
                            "width": 256,
                            "height": 256,
                        },
                        "rows": 1,
                        "columns": 1,
                    },
                ],
            },
        }
    )

    assert later_response["metadata"]["palette"] == {
        "2": "#46505A",
        "4": "#010203",
    }
    assert [
        view["samples"] for view in later_response["metadata"]["views"]
    ] == [["4"], ["2"]]


def test_read_pixels_is_unavailable_without_visual_observations(
    tmp_path: Path,
) -> None:
    controller, env, _, _ = make_controller(tmp_path, observation_mode="grid")

    response = controller.handle(
        {
            "method": "read_pixels",
            "arguments": {
                "question": "What is sampled?",
                "views": [
                    {
                        "label": "overview",
                        "turn": 0,
                        "region": {
                            "x": 0,
                            "y": 0,
                            "width": 512,
                            "height": 512,
                        },
                        "rows": 1,
                        "columns": 1,
                    }
                ],
            },
        }
    )

    assert response == {
        "ok": False,
        "metadata": {"error": "Visual observations are unavailable."},
    }
    assert env.calls == []


def test_read_pixels_accepts_more_views_than_image_inspection(
    tmp_path: Path,
) -> None:
    controller, env, _, _ = make_controller(tmp_path, render_scale=8)
    views = [
        {
            "label": f"sample {index}",
            "turn": 0,
            "region": {"x": index, "y": 0, "width": 1, "height": 1},
            "rows": 1,
            "columns": 1,
        }
        for index in range(19)
    ]

    response = controller.handle(
        {
            "method": "read_pixels",
            "arguments": {"question": "Read all selected pixels.", "views": views},
        }
    )

    assert response["ok"] is True
    assert response["metadata"]["view_count"] == 19
    assert response["metadata"]["sample_count"] == 19
    assert env.calls == []


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"question": "", "views": []},
        {
            "question": "q",
            "views": [
                {
                    "label": "v",
                    "turn": 0,
                    "region": {"x": 0, "y": 0, "width": 8, "height": 8},
                    "rows": True,
                    "columns": 1,
                }
            ],
        },
        {
            "question": "q",
            "views": [
                {
                    "label": "v",
                    "turn": 0,
                    "region": {"x": 0, "y": 0, "width": 8, "height": 8},
                    "rows": 0,
                    "columns": 1,
                }
            ],
        },
        {
            "question": "q",
            "views": [
                {
                    "label": "v",
                    "turn": 0,
                    "region": {"x": 511, "y": 0, "width": 2, "height": 1},
                    "rows": 1,
                    "columns": 1,
                }
            ],
        },
        {
            "question": "q",
            "views": [
                {
                    "label": "v",
                    "turn": 0,
                    "region": {
                        "x": 0,
                        "y": 0,
                        "width": 512,
                        "height": 512,
                    },
                    "rows": 65,
                    "columns": 64,
                }
            ],
        },
        {
            "question": "q",
            "views": [
                {
                    "label": "v",
                    "turn": 0,
                    "region": {"x": 0, "y": 0, "width": 1, "height": 1},
                    "rows": 1,
                    "columns": 1,
                }
            ]
            * 65,
        },
    ],
)
def test_read_pixels_rejects_malformed_or_excessive_requests(
    tmp_path: Path,
    arguments: dict,
) -> None:
    controller, env, _, _ = make_controller(tmp_path, render_scale=8)

    response = controller.handle({"method": "read_pixels", "arguments": arguments})

    assert response["ok"] is False
    assert "error" in response["metadata"]
    assert env.calls == []


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"question": "", "views": [{"label": "overview", "turn": 0}]},
        {"views": []},
        {"question": "q", "views": [{"turn": 0}]},
        {"question": "q", "views": [{"label": "", "turn": 0}]},
        {"question": "q", "views": [{"label": "v", "turn": True}]},
        {
            "question": "q",
            "views": [{"label": "v", "turn": 0, "frame": True}],
        },
        {
            "question": "q",
            "views": [{"label": "v", "turn": 0, "frame": 999}],
        },
        {
            "question": "q",
            "views": [{"label": "v", "turn": 0, "region": None}],
        },
        {
            "question": "q",
            "views": [{"label": "v", "turn": 0, "orientation": 90}],
        },
        {
            "question": "q",
            "views": [{"label": "v", "turn": 0, "unexpected": 1}],
        },
        {
            "question": "q",
            "views": [
                {
                    "label": "v",
                    "turn": 0,
                    "region": {"x": 511, "y": 0, "width": 2, "height": 1},
                }
            ]
        },
        {
            "question": "q",
            "views": [
                {
                    "label": "v",
                    "turn": 0,
                    "region": {"x": 0, "y": 0, "width": 0, "height": 1},
                }
            ]
        },
        {
            "question": "q",
            "views": [{"label": "v", "turn": 0}] * 17,
        },
    ],
)
def test_inspect_rejects_malformed_selection(
    tmp_path: Path,
    arguments: dict,
) -> None:
    controller, env, _, _ = make_controller(tmp_path, render_scale=8)

    response = controller.handle({"method": "inspect", "arguments": arguments})

    assert response["ok"] is False
    assert "error" in response["metadata"]
    assert env.calls == []


def test_history_exposes_only_public_actions_and_environment_results(
    tmp_path: Path,
) -> None:
    controller, env, _, _ = make_controller(
        tmp_path,
        observation_mode="both",
        render_scale=8,
        next_observations=[
            observation(1, state=GameState.GAME_OVER),
            observation(2),
            observation(3),
        ],
    )

    controller.handle(
        {
            "method": "play",
            "request_id": 1,
            "arguments": {"action": "ACTION6", "x": 300, "y": 100},
        }
    )
    reset = controller.handle(
        {
            "method": "play",
            "request_id": 2,
            "arguments": {
                "action": "RESET",
                "retry_state": "The first click causes GAME_OVER.",
            },
        }
    )
    assert reset["metadata"]["retry_boundary"] is True
    recovery = controller.begin_retry_recovery()
    assert recovery.boundary_step == 2
    assert recovery.current_observation["turn"] == 2
    assert len(recovery.image_paths) == 1
    controller.complete_retry_recovery(recovery)
    controller.handle(
        {
            "method": "play",
            "request_id": 3,
            "arguments": {"action": "ACTION1"},
        }
    )

    attempts = controller.handle(
        {"method": "history", "arguments": {"view": "attempts"}}
    )
    events = controller.handle(
        {
            "method": "history",
            "arguments": {
                "view": "events",
                "start_turn": 1,
                "end_turn": 2,
                "limit": 2,
            },
        }
    )

    assert attempts["ok"] is True
    assert attempts["metadata"]["attempts"] == [
        {
            "attempt": 1,
            "start": "level_entry",
            "start_turn": 0,
            "outcome": "GAME_OVER",
            "game_actions": 1,
            "end_turn": 1,
        },
        {
            "attempt": 2,
            "start": "reset",
            "start_turn": 2,
            "outcome": "NOT_FINISHED",
            "game_actions": 1,
            "end_turn": 3,
        },
    ]
    assert events["ok"] is True
    assert events["metadata"]["available_turn_range"] == [1, 3]
    assert events["metadata"]["total_matching"] == 2
    assert events["metadata"]["truncated"] is False
    first = events["metadata"]["events"][0]
    assert first["action"] == {"action": "ACTION6", "x": 300, "y": 100}
    assert "environment_x" not in first["action"]
    assert "environment_y" not in first["action"]
    assert "grid" not in first["result"]
    assert events["metadata"]["events"][1]["action"] == {"action": "RESET"}
    assert len(env.calls) == 3


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"view": "unknown"},
        {"view": "attempts", "start_turn": 0},
        {"view": "events", "limit": 0},
        {"view": "events", "start_turn": True},
        {"view": "events", "start_turn": 3, "end_turn": 2},
        {"view": "events", "unexpected": 1},
    ],
)
def test_history_rejects_malformed_queries(
    tmp_path: Path,
    arguments: dict,
) -> None:
    controller, env, _, _ = make_controller(tmp_path)

    response = controller.handle({"method": "history", "arguments": arguments})

    assert response["ok"] is False
    assert "error" in response["metadata"]
    assert env.calls == []


def test_invalid_retry_limit_reports_without_changing_game_state(
    tmp_path: Path,
) -> None:
    controller, env, _, _ = make_controller(tmp_path, max_invalid_retries=1)
    request_value = {
        "method": "play",
        "arguments": {"action": "ACTION3"},
    }

    first = controller.handle({**request_value, "request_id": 1})
    second = controller.handle({**request_value, "request_id": 2})
    third = controller.handle({**request_value, "request_id": 3})

    assert first["ok"] is False
    assert second["ok"] is False
    assert second["metadata"]["terminal"] is False
    assert second["metadata"]["invalid_action_limit"] is True
    assert second["metadata"]["error"] == "ACTION3 is not currently available."
    assert "termination_reason" not in second["metadata"]
    assert third["ok"] is False
    assert "invalid_action_limit" not in third["metadata"]
    assert controller.terminal is False
    assert controller.step_index == 0
    assert env.calls == []

    normal_execution = execution_from_response(second)
    assert normal_execution.success is False
    assert normal_execution.interrupt_after is False

    interrupting_execution = execution_from_response(
        second,
        interrupt_on_invalid_action_limit=True,
    )
    assert interrupting_execution.success is False
    assert interrupting_execution.interrupt_after is True
    assert interrupting_execution.boundary_reason == "invalid_action_limit"


def test_missing_environment_response_fails_closed_without_advancing(
    tmp_path: Path,
) -> None:
    controller, env, _, _ = make_controller(
        tmp_path,
        next_observations=[None],
    )

    with pytest.raises(RuntimeError, match="remote state is unknown"):
        controller.handle(
            {
                "method": "play",
                "request_id": 3,
                "arguments": {"action": "ACTION1"},
            }
        )

    assert len(env.calls) == 1
    assert controller.step_index == 0
    assert controller.terminal is True
    assert controller.termination_reason == "environment_response_missing"


def test_runtime_recovery_snapshots_state_without_game_action(tmp_path: Path) -> None:
    guide = tmp_path / "visible" / "GUIDE.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("Durable model.\n", encoding="utf-8")
    working = guide.with_name("WORKING.md")
    working.write_text("Current plan.\n", encoding="utf-8")
    controller, env, private, _ = make_controller(
        tmp_path,
        next_observations=[observation(3)],
        guide_path=guide,
        working_path=working,
    )
    controller.handle(
        {
            "method": "play",
            "request_id": 4,
            "arguments": {"action": "ACTION1"},
        }
    )
    calls_before = list(env.calls)

    recovery = controller.begin_runtime_recovery()
    controller.complete_runtime_recovery(recovery)

    assert recovery.boundary_step == 1
    assert recovery.guide == "Durable model.\n"
    assert recovery.working_memory == "Current plan.\n"
    assert recovery.objective_history["attempts"][-1]["game_actions"] == 1
    assert recovery.current_observation["turn"] == 1
    assert recovery.last_action_result is not None
    assert recovery.last_action_result["action"] == {"action": "ACTION1"}
    assert len(recovery.image_paths) == 1
    assert env.calls == calls_before
    assert controller.step_index == 1
    event = json.loads((private / "compact_recovery.jsonl").read_text())
    assert event["delivery"] == "resumed_runtime_thread"
    assert event["action_applied"] is False


def test_action_budget_becomes_terminal(tmp_path: Path) -> None:
    controller, _, _, _ = make_controller(tmp_path, max_steps=1)

    response = controller.handle(
        {"method": "play", "request_id": 1, "arguments": {"action": "ACTION1"}}
    )

    assert response["metadata"]["terminal"] is True
    assert response["metadata"]["termination_reason"] == "action_budget_exhausted"
    assert controller.total_attempts == 1


def test_game_over_exposes_only_reset_and_can_continue(tmp_path: Path) -> None:
    controller, env, _, _ = make_controller(
        tmp_path,
        next_observations=[
            observation(state=GameState.GAME_OVER),
            observation(4),
        ],
    )

    game_over = controller.handle(
        {"method": "play", "request_id": 1, "arguments": {"action": "ACTION1"}}
    )
    invalid = controller.handle(
        {"method": "play", "request_id": 2, "arguments": {"action": "ACTION1"}}
    )
    reset = controller.handle(
        {
            "method": "play",
            "request_id": 3,
            "arguments": {
                "action": "RESET",
                "retry_state": "The first action causes GAME_OVER.",
            },
        }
    )

    assert game_over["metadata"]["state"] == "GAME_OVER"
    assert game_over["metadata"]["terminal"] is False
    assert game_over["metadata"]["available_actions"] == ["RESET"]
    assert invalid["ok"] is False
    assert [call["method"] for call in env.calls] == ["step", "reset"]
    assert env.calls[0]["action"] == GameAction.ACTION1
    assert env.calls[1]["action"] == GameAction.RESET
    assert reset["metadata"]["state"] == "NOT_FINISHED"
    assert reset["metadata"]["terminal"] is False
    assert reset["metadata"]["available_actions"] == ["ACTION1", "ACTION6"]
    assert reset["metadata"]["retry_boundary"] is True
    assert controller.retry_boundary_pending is True
    assert controller.step_index == 2


def test_reset_boundary_replaces_working_and_blocks_actions_until_recovered(
    tmp_path: Path,
) -> None:
    working = tmp_path / "visible" / "WORKING.md"
    guide = tmp_path / "visible" / "GUIDE.md"
    recovery_dir = tmp_path / "private" / "codex_home"
    checkpoint = recovery_dir / "compact_checkpoint.requested"
    ready = recovery_dir / "compact_checkpoint.ready"
    retry_marker = recovery_dir / "retry_boundary.pending"
    working.parent.mkdir(parents=True)
    recovery_dir.mkdir(parents=True)
    working.write_text("Failed-attempt state.\n", encoding="utf-8")
    guide.write_text("Durable model.\n", encoding="utf-8")
    controller, env, _, _ = make_controller(
        tmp_path,
        next_observations=[
            observation(state=GameState.GAME_OVER),
            observation(4),
            observation(5),
        ],
        compact_checkpoint_marker=checkpoint,
        compact_checkpoint_ready=ready,
        retry_boundary_marker=retry_marker,
        working_path=working,
        guide_path=guide,
    )
    old_working_provenance = controller.record_working_write()

    controller.handle(
        {"method": "play", "request_id": 1, "arguments": {"action": "ACTION1"}}
    )
    reset = controller.handle(
        {
            "method": "play",
            "request_id": 2,
            "arguments": {
                "action": "RESET",
                "retry_state": "Next-attempt continuation.",
            },
        }
    )
    blocked = controller.handle(
        {"method": "play", "request_id": 3, "arguments": {"action": "ACTION1"}}
    )

    assert reset["metadata"]["retry_boundary"] is True
    assert working.read_text(encoding="utf-8") == "Next-attempt continuation.\n"
    assert retry_marker.is_file()
    assert guide.read_text(encoding="utf-8") == "Durable model.\n"
    assert blocked["ok"] is False
    assert blocked["metadata"]["error"] == (
        "Retry recovery has not been delivered yet."
    )
    assert len(env.calls) == 2

    checkpoint.touch()
    ready.touch()
    recovery = controller.begin_retry_recovery()
    assert not checkpoint.exists()
    assert not ready.exists()
    assert recovery.guide == "Durable model.\n"
    assert recovery.working_memory == "Next-attempt continuation.\n"
    assert recovery.working_provenance == {
        "source": "reset_handoff",
        "turn": 1,
        "attempt": 1,
        "state": "GAME_OVER",
        "progress": {"completed": 0, "total": 1},
        "level_start_turn": 0,
        "reset_turn": 2,
    }
    assert recovery.working_provenance != old_working_provenance
    assert recovery.last_action_result is not None
    assert recovery.last_action_result["action"]["action"] == "RESET"
    assert recovery.objective_history["attempts"][-1]["start"] == "reset"
    controller.complete_retry_recovery(recovery)
    assert working.read_text(encoding="utf-8") == "Next-attempt continuation.\n"
    continued = controller.handle(
        {"method": "play", "request_id": 4, "arguments": {"action": "ACTION1"}}
    )

    assert continued["metadata"]["action_applied"] is True
    assert controller.retry_boundary_pending is False
    assert not retry_marker.exists()
    assert len(env.calls) == 3
    recovery_events = [
        json.loads(line)
        for line in (tmp_path / "private" / "compact_recovery.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert recovery_events[-1] == {
        "step": 2,
        "action_applied": False,
        "delivery": "fresh_retry_thread",
    }


def test_terminal_reset_is_still_a_visual_boundary(tmp_path: Path) -> None:
    retry_marker = tmp_path / "private" / "codex_home" / "retry_boundary.pending"
    controller, _, _, _ = make_controller(
        tmp_path,
        max_steps=2,
        next_observations=[
            observation(state=GameState.GAME_OVER),
            observation(4),
        ],
        retry_boundary_marker=retry_marker,
    )

    controller.handle(
        {"method": "play", "request_id": 1, "arguments": {"action": "ACTION1"}}
    )
    reset = controller.handle(
        {
            "method": "play",
            "request_id": 2,
            "arguments": {
                "action": "RESET",
                "retry_state": "Preserve the failed attempt.",
            },
        }
    )

    assert reset["metadata"]["terminal"] is True
    assert reset["metadata"]["termination_reason"] == "action_budget_exhausted"
    assert reset["metadata"]["retry_boundary"] is True
    assert controller.retry_boundary_pending is False
    assert retry_marker.is_file()

    controller.discard_compact_recovery()
    assert not retry_marker.exists()


def test_win_remains_terminal(tmp_path: Path) -> None:
    controller, env, _, _ = make_controller(
        tmp_path,
        initial_observation=observation(state=GameState.WIN),
    )

    metadata = controller.initial_metadata()
    response = controller.handle(
        {"method": "play", "request_id": 1, "arguments": {"action": "RESET"}}
    )

    assert metadata["terminal"] is True
    assert metadata["termination_reason"] == "WIN"
    assert metadata["available_actions"] == []
    assert response["metadata"]["terminal"] is True
    assert env.calls == []


def test_level_boundary_stops_batch_but_next_call_continues(tmp_path: Path) -> None:
    controller, env, _, _ = make_controller(
        tmp_path,
        initial_observation=observation(levels_completed=0, win_levels=3),
        next_observations=[
            observation(1, levels_completed=1, win_levels=3),
            observation(2, levels_completed=1, win_levels=3),
        ],
    )

    crossed = controller.handle(
        {"method": "play", "request_id": 1, "arguments": {"action": "ACTION1"}}
    )
    continued = controller.handle(
        {"method": "play", "request_id": 2, "arguments": {"action": "ACTION1"}}
    )

    assert crossed["metadata"]["action_applied"] is True
    assert crossed["metadata"]["level_boundary"] is True
    assert "image" in crossed
    assert continued["metadata"]["action_applied"] is True
    assert "level_boundary" not in continued["metadata"]
    assert controller.step_index == 2
    assert controller.total_attempts == 2
    assert len(env.calls) == 2


def test_level_boundary_archives_and_clears_current_level_working_state(
    tmp_path: Path,
) -> None:
    working = tmp_path / "visible" / "WORKING.md"
    guide = tmp_path / "visible" / "GUIDE.md"
    working.parent.mkdir(parents=True)
    working.write_text("Temporary current-level state.\n", encoding="utf-8")
    guide.write_text("Durable cross-level model.\n", encoding="utf-8")
    controller, env, private, _ = make_controller(
        tmp_path,
        initial_observation=observation(levels_completed=0, win_levels=3),
        next_observations=[
            observation(1, levels_completed=1, win_levels=3),
            observation(2, levels_completed=1, win_levels=3),
        ],
        working_path=working,
        guide_path=guide,
    )

    response = controller.handle(
        {"method": "play", "arguments": {"action": "ACTION1"}}
    )

    assert response["metadata"]["level_boundary"] is True
    assert not working.exists()
    assert not (private / "working_memory_provenance.json").exists()
    assert guide.read_text(encoding="utf-8") == "Durable cross-level model.\n"
    archives = sorted((private / "working_memory_archive").glob("*.md"))
    assert len(archives) == 1
    assert archives[0].read_text(encoding="utf-8") == (
        "Temporary current-level state.\n"
    )
    archive_metadata = json.loads(archives[0].with_suffix(".json").read_text())
    assert archive_metadata["boundary_turn"] == 1
    assert archive_metadata["previous_progress"] == {
        "completed": 0,
        "total": 3,
    }
    assert archive_metadata["current_progress"] == {
        "completed": 1,
        "total": 3,
    }
    recovery = controller.begin_runtime_recovery()
    assert recovery.working_memory is None
    assert recovery.working_provenance is None

    dispatcher = GameToolDispatcher(
        controller=controller,
        guide_path=guide,
        working_path=working,
    )
    continued = dispatcher.execute(
        "play",
        {"action": "ACTION1"},
        "play-after-boundary",
    )
    current_working = dispatcher.execute(
        "read_working",
        {},
        "read-current-working",
    )

    assert continued.success is True
    assert len(env.calls) == 2
    assert current_working.success is True
    assert current_working.text == "WORKING.md does not exist."
    attempts = controller.handle(
        {"method": "history", "arguments": {"view": "attempts"}}
    )
    assert attempts["metadata"]["level_start_turn"] == 1
    assert attempts["metadata"]["attempts"][0]["game_actions"] == 1


@pytest.mark.parametrize(
    ("next_observation", "max_steps", "termination_reason"),
    [
        (observation(state=GameState.WIN, levels_completed=1), 10, "WIN"),
        (observation(levels_completed=1, win_levels=3), 1, "action_budget_exhausted"),
    ],
)
def test_terminal_progress_does_not_report_level_boundary(
    tmp_path: Path,
    next_observation,
    max_steps: int,
    termination_reason: str,
) -> None:
    controller, _, _, _ = make_controller(
        tmp_path,
        initial_observation=observation(levels_completed=0, win_levels=3),
        next_observations=[next_observation],
        max_steps=max_steps,
    )

    response = controller.handle(
        {"method": "play", "request_id": 1, "arguments": {"action": "ACTION1"}}
    )

    assert response["metadata"]["terminal"] is True
    assert response["metadata"]["termination_reason"] == termination_reason
    assert "level_boundary" not in response["metadata"]


def test_compact_recovery_moves_checkpoint_into_fresh_thread(tmp_path: Path) -> None:
    marker = tmp_path / "private" / "codex_home" / "compact_restore.pending"
    marker.parent.mkdir(parents=True)
    snapshot = marker.with_name("compact_guide.snapshot")
    checkpoint = marker.with_name("compact_checkpoint.requested")
    checkpoint.touch()
    ready = marker.with_name("compact_checkpoint.ready")
    ready.touch()
    guide = tmp_path / "visible" / "GUIDE.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("Durable model\n", encoding="utf-8")
    working = guide.with_name("WORKING.md")
    working.write_text("Selected source at turn 147.\n", encoding="utf-8")
    controller, env, private, _ = make_controller(
        tmp_path,
        next_observations=[observation(5)],
        compact_restore_marker=marker,
        compact_guide_snapshot=snapshot,
        compact_checkpoint_marker=checkpoint,
        compact_checkpoint_ready=ready,
        working_path=working,
        guide_path=guide,
    )

    recovery = controller.begin_compact_recovery()

    assert recovery.guide == "Durable model\n"
    assert recovery.working_memory == "Selected source at turn 147.\n"
    assert recovery.current_observation["action_applied"] is False
    assert recovery.last_action_result is None
    assert recovery.objective_history["attempts"] == [
        {
            "attempt": 1,
            "start": "level_entry",
            "start_turn": 0,
            "outcome": "NOT_FINISHED",
            "game_actions": 0,
            "end_turn": 0,
        }
    ]
    assert len(recovery.image_paths) == 1
    assert recovery.image_paths[0].is_file()
    assert marker.is_file()
    assert snapshot.read_text(encoding="utf-8") == "Durable model\n"

    controller.complete_compact_recovery(recovery)

    assert marker.exists() is False
    assert snapshot.exists() is False
    assert checkpoint.exists() is False
    assert ready.exists() is False
    assert working.read_text(encoding="utf-8") == "Selected source at turn 147.\n"
    assert guide.read_text(encoding="utf-8") == "Durable model\n"
    assert controller.step_index == 0
    assert controller.total_attempts == 0
    assert env.calls == []
    recovery = json.loads((private / "compact_recovery.jsonl").read_text())
    assert recovery["delivery"] == "fresh_thread"
    assert recovery["action_applied"] is False


def test_compact_recovery_includes_exact_last_event_and_attempt_history(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "private" / "codex_home" / "compact_restore.pending"
    marker.parent.mkdir(parents=True)
    snapshot = marker.with_name("compact_guide.snapshot")
    checkpoint = marker.with_name("compact_checkpoint.requested")
    ready = marker.with_name("compact_checkpoint.ready")
    guide = tmp_path / "visible" / "GUIDE.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("Durable model.\n", encoding="utf-8")
    working = guide.with_name("WORKING.md")
    working.write_text("Current state and intention.\n", encoding="utf-8")
    controller, _, _, _ = make_controller(
        tmp_path,
        observation_mode="both",
        render_scale=8,
        next_observations=[observation(4)],
        compact_restore_marker=marker,
        compact_guide_snapshot=snapshot,
        compact_checkpoint_marker=checkpoint,
        compact_checkpoint_ready=ready,
        working_path=working,
        guide_path=guide,
    )
    controller.handle(
        {
            "method": "play",
            "arguments": {"action": "ACTION6", "x": 300, "y": 100},
        }
    )
    checkpoint.touch()
    ready.touch()

    recovery = controller.begin_compact_recovery()

    assert recovery.current_observation["turn"] == 1
    assert recovery.last_action_result is not None
    assert recovery.last_action_result["action"] == {
        "action": "ACTION6",
        "x": 300,
        "y": 100,
    }
    assert recovery.last_action_result["result"]["turn"] == 1
    assert "grid" not in recovery.last_action_result["result"]
    assert recovery.objective_history["attempts"] == [
        {
            "attempt": 1,
            "start": "level_entry",
            "start_turn": 0,
            "outcome": "NOT_FINISHED",
            "game_actions": 1,
            "end_turn": 1,
        }
    ]


def test_compact_checkpoint_atomically_updates_guide_and_working_memory(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "private" / "codex_home" / "compact_checkpoint.requested"
    ready = checkpoint.with_name("compact_checkpoint.ready")
    working = tmp_path / "visible" / "WORKING.md"
    guide = tmp_path / "visible" / "GUIDE.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text("Old durable model.\n", encoding="utf-8")
    controller, env, _, _ = make_controller(
        tmp_path,
        compact_checkpoint_marker=checkpoint,
        compact_checkpoint_ready=ready,
        working_path=working,
        guide_path=guide,
    )

    rejected = controller.handle(
        {
            "method": "save_compact_checkpoint",
            "request_id": 1,
            "arguments": {
                "guide": "New durable model.",
                "working_memory": "Local state",
            },
        }
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    saved = controller.handle(
        {
            "method": "save_compact_checkpoint",
            "request_id": 2,
            "arguments": {
                "guide": "  New durable model.  ",
                "working_memory": "  Local state at turn 9.  ",
            },
        }
    )
    blocked_play = controller.handle(
        {"method": "play", "request_id": 3, "arguments": {"action": "ACTION1"}}
    )
    duplicate = controller.handle(
        {
            "method": "save_compact_checkpoint",
            "request_id": 4,
            "arguments": {
                "guide": "Must not replace the committed guide.",
                "working_memory": "Must not replace committed working memory.",
            },
        }
    )

    assert rejected["ok"] is False
    assert saved["metadata"] == {
        "checkpoint_saved": True,
        "guide_updated": True,
        "guide_chars": 18,
        "working_chars": 22,
    }
    assert guide.read_text(encoding="utf-8") == "New durable model.\n"
    assert working.read_text(encoding="utf-8") == "Local state at turn 9.\n"
    assert ready.is_file()
    assert duplicate["metadata"] == {
        "checkpoint_saved": True,
        "already_saved": True,
    }
    assert guide.read_text(encoding="utf-8") == "New durable model.\n"
    assert working.read_text(encoding="utf-8") == "Local state at turn 9.\n"
    assert blocked_play["ok"] is False
    assert blocked_play["metadata"]["action_applied"] is False
    assert env.calls == []


def test_compact_checkpoint_can_preserve_existing_guide(tmp_path: Path) -> None:
    checkpoint = tmp_path / "private" / "codex_home" / "compact_checkpoint.requested"
    ready = checkpoint.with_name("compact_checkpoint.ready")
    working = tmp_path / "visible" / "WORKING.md"
    guide = tmp_path / "visible" / "GUIDE.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text("Current durable model.\n", encoding="utf-8")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()
    controller, _, _, _ = make_controller(
        tmp_path,
        compact_checkpoint_marker=checkpoint,
        compact_checkpoint_ready=ready,
        working_path=working,
        guide_path=guide,
    )

    saved = controller.handle(
        {
            "method": "save_compact_checkpoint",
            "arguments": {
                "guide": None,
                "working_memory": "Current selection at turn 12.",
            },
        }
    )

    assert saved["ok"] is True
    assert saved["metadata"]["guide_updated"] is False
    assert guide.read_text(encoding="utf-8") == "Current durable model.\n"
    assert working.read_text(encoding="utf-8") == "Current selection at turn 12.\n"
    assert ready.is_file()


def test_compact_checkpoint_failure_never_marks_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "private" / "codex_home" / "compact_checkpoint.requested"
    ready = checkpoint.with_name("compact_checkpoint.ready")
    working = tmp_path / "visible" / "WORKING.md"
    guide = tmp_path / "visible" / "GUIDE.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text("Old durable model.\n", encoding="utf-8")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()
    controller, _, _, _ = make_controller(
        tmp_path,
        compact_checkpoint_marker=checkpoint,
        compact_checkpoint_ready=ready,
        working_path=working,
        guide_path=guide,
    )
    real_atomic_write = controller_module.atomic_write_text

    def fail_working_write(path: Path, content: str, *, mode: int) -> None:
        if path == working:
            raise OSError("simulated write failure")
        real_atomic_write(path, content, mode=mode)

    monkeypatch.setattr(controller_module, "atomic_write_text", fail_working_write)
    failed = controller.handle(
        {
            "method": "save_compact_checkpoint",
            "arguments": {
                "guide": "New durable model.",
                "working_memory": "Current selection at turn 12.",
            },
        }
    )

    assert failed["ok"] is False
    assert not working.exists()
    assert not ready.exists()


def test_compact_recovery_fails_closed_without_working_memory(tmp_path: Path) -> None:
    marker = tmp_path / "private" / "codex_home" / "compact_restore.pending"
    marker.parent.mkdir(parents=True)
    snapshot = marker.with_name("compact_guide.snapshot")
    checkpoint = marker.with_name("compact_checkpoint.requested")
    checkpoint.touch()
    ready = marker.with_name("compact_checkpoint.ready")
    ready.touch()
    guide = tmp_path / "visible" / "GUIDE.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("Current model\n", encoding="utf-8")
    working = guide.with_name("WORKING.md")
    controller, env, _, _ = make_controller(
        tmp_path,
        compact_restore_marker=marker,
        compact_guide_snapshot=snapshot,
        compact_checkpoint_marker=checkpoint,
        compact_checkpoint_ready=ready,
        working_path=working,
        guide_path=guide,
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        controller.begin_compact_recovery()

    assert controller.terminal is True
    assert controller.termination_reason == "compact_recovery_failed"
    assert env.calls == []
