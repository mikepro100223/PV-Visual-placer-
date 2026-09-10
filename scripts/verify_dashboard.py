"""Exercise the actual dashboard with real geodata, checkpoints and weather.

This is an opt-in integration check, separate from offline unit tests. It uses
Streamlit's application runner, not a browser: visual layout is not validated.
Public geodata requests can be made; no data is uploaded or published.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from rooftop_pv.runtime import ROOT


def assert_healthy(app):
    errors = [entry.message for entry in app.exception]
    errors.extend(entry.value for entry in app.error)
    if errors:
        raise RuntimeError("Dashboard check failed: " + " | ".join(errors))


def click(app, label):
    button = next(button for button in app.button if button.label == label)
    if button.disabled:
        raise RuntimeError(f"Required dashboard action is disabled: {label}")
    button.click().run(timeout=120)
    assert_healthy(app)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="Hauptstrasse 1 5200 Brugg")
    parser.add_argument("--require-obstacles", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/demo/dashboard-check.json")
    args = parser.parse_args()

    app = AppTest.from_file(ROOT / "code/app.py", default_timeout=120).run()
    assert_healthy(app)
    next(field for field in app.text_input if field.label == "Schweizer Adresse").set_value(args.address)
    click(app, "Adresse suchen")
    click(app, "Dachfläche und Orthofoto laden")
    click(app, "PV-Segmentierung ausführen")
    obstacle_button = next(button for button in app.button if button.label == "Hindernis-Segmentierung ausführen")
    if args.require_obstacles or not obstacle_button.disabled:
        click(app, "Hindernis-Segmentierung ausführen")
    click(app, "PVGIS TMY laden und Szenario berechnen")

    state = app.session_state.filtered_state
    scenario = state["scenario"]
    summary = scenario["summary"]
    assert state["pv_inference_run"] is True
    assert len(scenario["monthly"]) == 12
    assert len(scenario["hourly"]) == 8760
    assert np.isfinite(summary["energy_kwh"])
    assert summary["usable_roof_plane_area_m2"] <= summary["gross_roof_plane_area_m2"]
    assert np.isclose(scenario["monthly"]["Energie (kWh)"].sum(), summary["energy_kwh"])
    for role in ("pv", "obstacles"):
        model = scenario["model"][role]
        if model["inference_run"]:
            assert model["weights"]["sha256"]
            assert model["image"]["sha256"] == scenario["provenance"]["orthophoto"]["sha256"]
    if args.require_obstacles:
        assert state["obstacle_inference_run"] is True

    report = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "scope": "Real dashboard actions through Streamlit AppTest; not browser-layout or field-accuracy validation.",
        "address": args.address,
        "summary": summary,
        "assumptions": scenario["assumptions"],
        "model": scenario["model"],
        "provenance": scenario["provenance"],
        "weather_metadata": scenario["weather_metadata"],
        "monthly_rows": len(scenario["monthly"]),
        "hourly_rows": len(scenario["hourly"]),
        "ui_warnings": [warning.value for warning in app.warning],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"passed": True, "report": str(args.output), "energy_kwh": summary["energy_kwh"]}))


if __name__ == "__main__":
    main()
