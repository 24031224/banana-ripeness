from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from banana.common import ConfigError
from banana.config import DEFAULT_THRESHOLDS_PATH, Thresholds, load_thresholds

PENDIENTE = pytest.mark.xfail(raises=NotImplementedError, reason="Esqueleto: pendiente")


# --- Pruebas que pasan desde el día 1: el YAML existe y respeta la escala de OpenCV ---


def test_yaml_has_all_sections() -> None:
    raw = yaml.safe_load(DEFAULT_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    assert {"version", "preprocess", "segmentation", "features", "classifier"} <= raw.keys()


def test_yaml_hue_values_use_opencv_scale() -> None:
    """OpenCV usa H en 0-179: ningún rango del YAML puede pasarse de 179."""
    raw = yaml.safe_load(DEFAULT_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    ranges = [raw["segmentation"]["banana_hsv"]] + [
        raw["features"][k] for k in ("green", "yellow", "brown")
    ]
    for r in ranges:
        assert 0 <= r["lower"][0] <= r["upper"][0] <= 179


# --- Pruebas del cargador (fallan con NotImplementedError hasta implementarlo) ---


@PENDIENTE
def test_load_default_thresholds() -> None:
    cfg = load_thresholds()
    assert isinstance(cfg, Thresholds)
    assert cfg.segmentation.banana_hsv.upper[0] <= 179


@PENDIENTE
def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_thresholds(tmp_path / "no_existe.yaml")


@PENDIENTE
def test_hue_over_179_raises_config_error(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    raw["features"]["green"]["upper"][0] = 200  # escala 0-360 por error
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_thresholds(bad)
