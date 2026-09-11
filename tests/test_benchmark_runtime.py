from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_runtime.py"
SPEC = importlib.util.spec_from_file_location("benchmark_runtime", SCRIPT)
assert SPEC and SPEC.loader
benchmark_runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_runtime)


def test_core_benchmark_is_reproducible_and_offline():
    report = benchmark_runtime.benchmark_core(hours=24, repeat=1)

    assert set(report) == {"geometry_and_placement", "pv_physics"}
    assert report["geometry_and_placement"]["runs"] == 1
    assert report["pv_physics"]["median_ms"] >= 0
