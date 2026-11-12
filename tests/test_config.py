from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from banana.common import ConfigError
from banana.config import DEFAULT_THRESHOLDS_PATH, Thresholds, load_thresholds

# --- El YAML existe y respeta la escala de OpenCV ---


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


# --- Pruebas del cargador ---


def test_load_default_thresholds() -> None:
    cfg = load_thresholds()
    assert isinstance(cfg, Thresholds)
    assert cfg.segmentation.banana_hsv.upper[0] <= 179


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_thresholds(tmp_path / "no_existe.yaml")


def test_hue_over_179_raises_config_error(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    raw["features"]["green"]["upper"][0] = 200  # escala 0-360 por error
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_thresholds(bad)


def _raw() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(DEFAULT_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    return copy.deepcopy(raw)


def _write(tmp_path: Path, raw: Any) -> Path:  # noqa: ANN401
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_default_path_is_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_thresholds().preprocess.target_width > 0


def test_loaded_values_match_yaml() -> None:
    raw = _raw()
    cfg = load_thresholds(str(DEFAULT_THRESHOLDS_PATH))
    assert cfg.version == raw["version"]
    assert cfg.segmentation.banana_hsv.lower == tuple(raw["segmentation"]["banana_hsv"]["lower"])
    assert cfg.features.brown.upper == tuple(raw["features"]["brown"]["upper"])
    assert cfg.classifier.inmaduro_green_ratio_min == raw["classifier"]["inmaduro_green_ratio_min"]
    assert cfg.preprocess.white_balance == raw["preprocess"]["white_balance"]


def test_invalid_yaml_syntax(tmp_path: Path) -> None:
    path = tmp_path / "roto.yaml"
    path.write_text("a: [1, 2\nb: : :", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_thresholds(path)


def test_root_not_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapa"):
        load_thresholds(_write(tmp_path, [1, 2, 3]))


@pytest.mark.parametrize("section", ["preprocess", "segmentation", "features", "classifier"])
def test_missing_section(tmp_path: Path, section: str) -> None:
    raw = _raw()
    del raw[section]
    with pytest.raises(ConfigError, match=section):
        load_thresholds(_write(tmp_path, raw))


def test_missing_key_message_names_the_key(tmp_path: Path) -> None:
    raw = _raw()
    del raw["features"]["dark_spot_value_max"]
    with pytest.raises(ConfigError, match=r"features\.dark_spot_value_max"):
        load_thresholds(_write(tmp_path, raw))


def _set(path: tuple[str, ...], value: Any) -> Callable[[dict[str, Any]], None]:  # noqa: ANN401
    def apply(raw: dict[str, Any]) -> None:
        node = raw
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return apply


@pytest.mark.parametrize(
    "mutate",
    [
        _set(("segmentation", "banana_hsv", "upper"), [200, 255, 255]),  # H fuera de 0-179
        _set(("features", "yellow", "lower"), [-1, 60, 100]),
        _set(("features", "green", "upper"), [85, 300, 255]),  # S fuera de 0-255
        _set(("features", "brown", "lower"), [30, 40, 30]),  # lower > upper en H
        _set(("features", "green", "lower"), [36, 40]),  # terna incompleta
        _set(("features", "green", "lower"), [36.5, 40, 40]),  # no entero
        _set(("preprocess", "blur_kernel"), 4),  # par
        _set(("segmentation", "morph_kernel"), 0),
        _set(("preprocess", "white_balance"), "magia"),
        _set(("preprocess", "target_width"), True),
        _set(("segmentation", "min_area_ratio"), 1.5),
        _set(("classifier", "no_comestible_brown_ratio_min"), -0.1),
        _set(("features", "hue_bins"), 0),
        _set(("features", "dark_spot_value_max"), 300),
        _set(("features", "green"), [1, 2, 3]),  # no es un mapa
        _set(("version",), "uno"),
    ],
)
def test_invalid_values_raise_config_error(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    raw = _raw()
    mutate(raw)
    with pytest.raises(ConfigError):
        load_thresholds(_write(tmp_path, raw))
