"""Provider-neutral execution of the public visual-game tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tools import MAX_GUIDE_CHARS, MAX_WORKING_CHARS


@dataclass(frozen=True)
class ToolExecution:
    text: str
    success: bool
    images: tuple[dict[str, Any], ...] = ()
    interrupt_after: bool = False
    boundary_reason: str | None = None


class GameToolDispatcher:
    """Execute the shared public tools against one private controller."""

    def __init__(
        self,
        *,
        controller: Any,
        guide_path: Path,
        working_path: Path,
        checkpoint_via_working: bool = False,
        interrupt_on_invalid_action_limit: bool = False,
    ) -> None:
        self.controller = controller
        self.guide_path = guide_path
        self.working_path = working_path
        self.checkpoint_via_working = checkpoint_via_working
        self.interrupt_on_invalid_action_limit = interrupt_on_invalid_action_limit

    def execute(
        self,
        tool: str,
        arguments: Any,
        call_id: str,
    ) -> ToolExecution:
        restore_pending = bool(
            self.controller.compact_restore_marker is not None
            and self.controller.compact_restore_marker.exists()
        )
        checkpoint_pending = bool(
            self.controller.compact_checkpoint_marker is not None
            and self.controller.compact_checkpoint_marker.exists()
        )
        if getattr(self.controller, "retry_boundary_pending", False):
            return ToolExecution("Retry recovery is being delivered.", False)
        if restore_pending:
            return ToolExecution("Compact recovery is being delivered.", False)
        if checkpoint_pending and not restore_pending:
            allowed_checkpoint_tools = (
                {"read_working", "write_working"}
                if self.checkpoint_via_working
                else {"save_compact_checkpoint"}
            )
            if tool not in allowed_checkpoint_tools:
                return ToolExecution(
                    (
                        "Review and update WORKING.md so it preserves relevant "
                        "existing information and contains the complete continuation "
                        "state needed after compaction."
                        if self.checkpoint_via_working
                        else "Save the compact checkpoint before using another tool."
                    ),
                    False,
                )

        if not isinstance(arguments, dict):
            arguments = {}
        if tool == "read_guide":
            return self._read_guide(arguments)
        if tool == "read_working":
            return self._read_working(arguments)
        if tool == "write_guide":
            return self._write_guide(arguments)
        if tool == "write_working":
            return self._write_working(arguments)
        if tool in {
            "save_compact_checkpoint",
            "inspect",
            "read_pixels",
            "history",
            "play",
        }:
            response = self.controller.handle(
                {
                    "method": tool,
                    "request_id": call_id,
                    "arguments": arguments,
                }
            )
            return execution_from_response(
                response,
                interrupt_on_invalid_action_limit=(
                    self.interrupt_on_invalid_action_limit
                ),
            )
        return ToolExecution("Unknown tool.", False)

    def _read_guide(self, arguments: dict[str, Any]) -> ToolExecution:
        if arguments:
            return ToolExecution("Invalid arguments.", False)
        try:
            return ToolExecution(
                self.guide_path.read_text(encoding="utf-8", errors="replace"),
                True,
            )
        except OSError:
            return ToolExecution("GUIDE.md is unavailable.", False)

    def _read_working(self, arguments: dict[str, Any]) -> ToolExecution:
        if arguments:
            return ToolExecution("Invalid arguments.", False)
        try:
            return ToolExecution(
                self.working_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ),
                True,
            )
        except FileNotFoundError:
            return ToolExecution("WORKING.md does not exist.", True)
        except OSError:
            return ToolExecution("WORKING.md is unavailable.", False)

    def _write_guide(self, arguments: dict[str, Any]) -> ToolExecution:
        content = arguments.get("content")
        if (
            set(arguments) != {"content"}
            or not isinstance(content, str)
            or not content.strip()
            or len(content) > MAX_GUIDE_CHARS
        ):
            return ToolExecution("Invalid GUIDE.md content.", False)
        try:
            atomic_write_text(self.guide_path, content.strip() + "\n")
        except OSError:
            return ToolExecution("GUIDE.md could not be updated.", False)
        return ToolExecution("GUIDE.md updated.", True)

    def _write_working(self, arguments: dict[str, Any]) -> ToolExecution:
        content = arguments.get("content")
        checkpoint_pending = bool(
            self.checkpoint_via_working
            and self.controller.compact_checkpoint_marker is not None
            and self.controller.compact_checkpoint_marker.exists()
        )
        if (
            set(arguments) != {"content"}
            or not isinstance(content, str)
            or not content.strip()
            or len(content) > MAX_WORKING_CHARS
        ):
            return ToolExecution("Invalid WORKING.md content.", False)
        try:
            atomic_write_text(self.working_path, content.strip() + "\n")
            record_method = getattr(self.controller, "record_working_write", None)
            if callable(record_method):
                record_method()
            if checkpoint_pending:
                complete_method = getattr(
                    self.controller,
                    "complete_compact_checkpoint_from_working",
                    None,
                )
                if not callable(complete_method):
                    raise RuntimeError(
                        "Compact checkpoint completion is unavailable."
                    )
                complete_method()
        except (OSError, RuntimeError):
            return ToolExecution("WORKING.md could not be updated.", False)
        if checkpoint_pending:
            return ToolExecution(
                "WORKING.md updated; compact checkpoint ready.",
                True,
                interrupt_after=True,
                boundary_reason="compact_checkpoint_saved",
            )
        return ToolExecution("WORKING.md updated.", True)


def execution_from_response(
    response: dict[str, Any],
    *,
    interrupt_on_invalid_action_limit: bool = False,
) -> ToolExecution:
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {"error": "Controller returned no metadata."}
    execution_images: list[dict[str, Any]] = []
    image = response.get("image")
    if isinstance(image, dict) and isinstance(image.get("data"), str):
        execution_images.append({**image, "steer_text": visual_result_label(metadata)})

    images = response.get("images")
    views = metadata.get("views")
    if isinstance(images, list) and isinstance(views, list):
        if len(images) != len(views):
            return _controller_image_error(
                "Controller returned mismatched inspection views."
            )
        for index, (view, view_image) in enumerate(
            zip(views, images, strict=True), start=1
        ):
            if not isinstance(view, dict) or not isinstance(view_image, dict):
                return _controller_image_error(
                    "Controller returned an invalid inspection view."
                )
            if not isinstance(view_image.get("data"), str):
                return _controller_image_error(
                    "Controller returned an invalid inspection image."
                )
            view_metadata = {
                "visual_kind": "inspection",
                "current_state_unchanged": True,
                "view_count": len(views),
                "index": index,
                **view,
            }
            execution_images.append(
                {
                    **view_image,
                    "steer_text": visual_result_label(view_metadata),
                }
            )
    invalid_action_limit = (
        interrupt_on_invalid_action_limit
        and metadata.get("invalid_action_limit") is True
    )
    return ToolExecution(
        json.dumps(metadata, separators=(",", ":")),
        response.get("ok") is True,
        tuple(execution_images),
        metadata.get("retry_boundary") is True or invalid_action_limit,
        "invalid_action_limit" if invalid_action_limit else None,
    )


def visual_result_label(metadata: dict[str, Any]) -> str:
    if metadata.get("visual_kind") == "inspection":
        attributes = ""
        index = metadata.get("index")
        view_count = metadata.get("view_count")
        if type(index) is int:
            attributes += f' index="{index}"'
        if type(view_count) is int:
            attributes += f' count="{view_count}"'
        attributes += (
            f' turn="{metadata.get("turn")}" frame="{metadata.get("frame")}"'
        )
        region = metadata.get("region")
        if isinstance(region, dict):
            attributes += (
                f' region="{region.get("x")},{region.get("y")},'
                f'{region.get("width")},{region.get("height")}"'
            )
        image_size = metadata.get("image_size")
        if (
            isinstance(image_size, dict)
            and type(image_size.get("width")) is int
            and type(image_size.get("height")) is int
        ):
            attributes += (
                f' size="{image_size["width"]}x{image_size["height"]}"'
            )
        return f"<inspection_visual{attributes}></inspection_visual>"

    visual = metadata.get("visual")
    if not isinstance(visual, dict):
        visual = {}
    turn = visual.get("turn", metadata.get("turn"))
    frame = visual.get("final_frame")
    frame_count = visual.get("frame_count")
    attributes = ""
    detail = visual.get("detail")
    width = visual.get("width")
    height = visual.get("height")
    if isinstance(detail, str):
        attributes += f' detail="{detail}"'
    if type(width) is int and type(height) is int:
        attributes += f' size="{width}x{height}"'
    if metadata.get("recovery_observation") is True:
        meaning = "Current game state restored after context recovery."
    elif metadata.get("action_applied") is True:
        meaning = "Current game state after the action."
    else:
        meaning = "Current game state."
    return (
        f'<current_visual turn="{turn}" frame="{frame}" '
        f'frame_count="{frame_count}"{attributes}>{meaning}</current_visual>'
    )


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _controller_image_error(message: str) -> ToolExecution:
    return ToolExecution(json.dumps({"error": message}, separators=(",", ":")), False)
