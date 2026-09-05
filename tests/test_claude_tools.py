import pytest

from vista_arc3.claude.tools import (
    MAX_INSPECTION_LABEL_CHARS,
    MAX_INSPECTION_QUESTION_CHARS,
    MAX_INSPECTION_VIEWS,
    MAX_PIXEL_READOUT_SAMPLES,
    MAX_PIXEL_READOUT_VIEWS,
    build_tools,
)


def test_tools_expose_only_the_visual_player_contract() -> None:
    tools = build_tools(512)

    assert [tool["name"] for tool in tools] == [
        "play",
        "inspect",
        "read_pixels",
        "history",
        "read_guide",
        "write_guide",
        "read_working",
        "write_working",
    ]
    by_name = {tool["name"]: tool for tool in tools}
    play = by_name["play"]
    action = play["inputSchema"]
    assert action["properties"]["x"]["maximum"] == 511
    assert action["properties"]["y"]["maximum"] == 511
    assert action["required"] == ["action"]
    assert set(action["properties"]) == {"action", "x", "y"}
    assert "Execute exactly one game action" in play["description"]
    assert "Each executed play call counts as one game action" in play["description"]
    assert "before another action can execute" in play["description"]
    assert "retry_state" not in play["description"]
    assert "Batch" not in play["description"]
    assert "usually associated with up, down, left, and right" in play["description"]
    assert "attach, detach, execute, etc." in play["description"]
    assert "semantically mapped" not in play["description"]

    inspect = by_name["inspect"]
    schema = inspect["inputSchema"]
    assert schema["required"] == ["question", "views"]
    question = schema["properties"]["question"]
    assert question["minLength"] == 1
    assert question["maxLength"] == MAX_INSPECTION_QUESTION_CHARS
    views = schema["properties"]["views"]
    assert views["minItems"] == 1
    assert views["maxItems"] == MAX_INSPECTION_VIEWS
    selection = views["items"]
    assert selection["required"] == ["label", "turn"]
    assert set(selection["properties"]) == {"label", "turn", "frame", "region"}
    assert selection["properties"]["label"]["maxLength"] == (
        MAX_INSPECTION_LABEL_CHARS
    )
    region = selection["properties"]["region"]
    assert region["properties"]["x"]["maximum"] == 511
    assert region["properties"]["width"]["maximum"] == 512
    assert "same standard 512x512 coordinates" in inspect["description"]
    assert "cropped exactly" in inspect["description"]
    assert "enlarged proportionally to fit within 512x512" in inspect["description"]
    assert "without smoothing" in inspect["description"]
    assert "question and labels remain in the inspect call" in inspect["description"]
    assert "rotation" not in inspect["description"].lower()

    read_pixels = by_name["read_pixels"]
    pixel_schema = read_pixels["inputSchema"]
    assert pixel_schema["required"] == ["question", "views"]
    pixel_view = pixel_schema["properties"]["views"]["items"]
    assert pixel_view["required"] == [
        "label",
        "turn",
        "region",
        "rows",
        "columns",
    ]
    assert set(pixel_view["properties"]) == {
        "label",
        "turn",
        "frame",
        "region",
        "rows",
        "columns",
    }
    assert pixel_view["properties"]["rows"]["maximum"] == 512
    assert pixel_view["properties"]["columns"]["maximum"] == 512
    assert (
        pixel_schema["properties"]["views"]["maxItems"]
        == MAX_PIXEL_READOUT_VIEWS
    )
    assert f"At most {MAX_PIXEL_READOUT_VIEWS} views" in read_pixels[
        "description"
    ]
    assert f"at most {MAX_PIXEL_READOUT_SAMPLES} samples" in read_pixels[
        "description"
    ]
    assert "archived supplied PNG" in read_pixels["description"]
    assert "one RGB symbol palette" in read_pixels["description"]
    assert "compact row string" in read_pixels["description"]
    assert "does not align, transform, compare, or interpret" in read_pixels[
        "description"
    ]

    history = by_name["history"]
    assert history["inputSchema"]["required"] == ["view"]
    assert history["inputSchema"]["properties"]["view"]["enum"] == [
        "attempts",
        "events",
    ]
    assert history["inputSchema"]["properties"]["limit"]["maximum"] == 128
    assert "objective action and environment-result records" in history["description"]

    assert by_name["read_guide"]["description"] == "Read `GUIDE.md`."
    assert by_name["write_guide"]["description"] == (
        "Write the complete contents of `GUIDE.md`, replacing its previous contents."
    )
    assert by_name["read_working"]["description"] == (
        "Read the agent-authored temporary state for the current level."
    )
    assert by_name["write_working"]["description"] == (
        "Replace the agent-authored temporary state for the current level. It "
        "persists across RESET and is cleared when level progress advances."
    )
    assert by_name["write_working"]["inputSchema"]["properties"]["content"][
        "type"
    ] == "string"


def test_same_session_reset_has_no_continuation_argument() -> None:
    tools = build_tools(
        512,
        include_compact_checkpoint=False,
        reset_requires_retry_state=False,
    )
    play = {tool["name"]: tool for tool in tools}["play"]

    assert set(play["inputSchema"]["properties"]) == {"action", "x", "y"}
    assert "retry_state" not in play["description"]
    assert "Each executed play call counts as one game action" in play["description"]


def test_tools_reject_invalid_display_size() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_tools(0)
