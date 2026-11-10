"""Select a benchmark profile without changing its evaluation arguments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=False, allow_abbrev=False
    )
    parser.add_argument(
        "--profile",
        choices=("arc3", "gameworld", "aigamestore", "babyvision"),
        required=True,
    )
    parser.add_argument(
        "--backend", choices=("codex", "claude", "dsh"), required=True
    )
    if not arguments or arguments in (["--help"], ["-h"]):
        parser.print_help()
        return 0
    selection, remaining = parser.parse_known_args(arguments)
    if selection.profile != "arc3" and selection.backend != "codex":
        parser.error(f"--profile {selection.profile} requires --backend codex")
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    if selection.profile == "arc3":
        from .benchmarks.arc3 import backend_module

        return backend_module(selection.backend).main(remaining)
    if selection.profile == "babyvision":
        from .benchmarks.babyvision.cli import main as babyvision_main

        return babyvision_main(["--backend", selection.backend, *remaining])
    if selection.profile == "aigamestore":
        from .benchmarks.aigamestore.cli import main as aigamestore_main

        return aigamestore_main(["--backend", selection.backend, *remaining])
    from .benchmarks.gameworld.cli import main as gameworld_main

    return gameworld_main(["--backend", selection.backend, *remaining])
