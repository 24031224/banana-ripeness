"""Pruebas del contrato compartido (estas SÍ deben pasar desde el día 1)."""

from __future__ import annotations

import dataclasses

import pytest

from banana.common import (
    DATASET_FOLDERS,
    BananaError,
    ConfigError,
    FeatureSet,
    ImageLoadError,
    Ripeness,
    SegmentationError,
    Verdict,
)


def _features() -> FeatureSet:
    return FeatureSet(
        hue_mean=27.0,
        hue_std=3.0,
        saturation_mean=200.0,
        value_mean=230.0,
        green_ratio=0.05,
        yellow_ratio=0.9,
        brown_ratio=0.05,
        dark_spot_ratio=0.0,
        hue_histogram=(0.0, 0.0, 1.0),
    )


def test_ripeness_labels_are_exact() -> None:
    assert [r.value for r in Ripeness] == ["Inmaduro", "Comestible", "No comestible"]


def test_dataset_folders_cover_all_classes() -> None:
    assert set(DATASET_FOLDERS.values()) == set(Ripeness)


def test_dataclasses_are_frozen() -> None:
    fs = _features()
    with pytest.raises(dataclasses.FrozenInstanceError):
        fs.green_ratio = 1.0  # type: ignore[misc]


def test_other_ratio_defaults_to_zero() -> None:
    assert _features().other_ratio == 0.0


def test_verdict_holds_features() -> None:
    fs = _features()
    v = Verdict(label=Ripeness.COMESTIBLE, confidence=0.8, reasons=("ok",), features=fs)
    assert v.features is fs


@pytest.mark.parametrize("exc", [ImageLoadError, SegmentationError, ConfigError])
def test_exceptions_share_base(exc: type[Exception]) -> None:
    assert issubclass(exc, BananaError)
