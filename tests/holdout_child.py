"""A construction process for the controller's tests: START or RESUME of the stand-in
construction (tests/holdout_fakes.py) on a directory, in a process of its own, as the
launcher's command would run it. No model runs.

    python -m tests.holdout_child <directory> start|resume '<scenario as JSON>'

The scenario's (cell, role) keys are written "cell|role"."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ingest import holdout_pipeline as hpl
from ingest import holdout_process as hproc
from tests.holdout_fakes import Scenario, load_base, make, reopen

PAIRED = ("records", "problem", "drop", "model", "override")


def scenario(data: dict[str, Any]) -> Scenario:
    fields = dict(data)
    for name in PAIRED:
        if name in fields:
            fields[name] = {tuple(k.split("|")): v for k, v in fields[name].items()}
    return Scenario(**fields)


def main(argv: list[str]) -> int:
    directory, action, data = Path(argv[0]), argv[1], json.loads(argv[2])
    hproc.install_termination()
    base = load_base()
    if action == "start":
        pipeline, _ = make(base, directory, scenario(data))
        pipeline.build()
    else:
        pipeline, _ = reopen(base, directory, scenario(data))
        pipeline.resume()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (hpl.TechnicalStop, hpl.ConstructionStop) as stop:
        print(type(stop).__name__, flush=True)
        sys.exit(1)
