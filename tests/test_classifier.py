from __future__ import annotations

import pytest

from banana.classifier import classify, margin_confidence
from banana.common import FeatureSet, Ripeness
from banana.config import Thresholds

pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="Esqueleto: pendiente")


def _fs(
    green: float = 0.0, yellow: float = 0.0, brown: float = 0.0, dark: float = 0.0
) -> FeatureSet:
    return FeatureSet(
        hue_mean=27.0,
        hue_std=5.0,
        saturation_mean=180.0,
        value_mean=200.0,
        green_ratio=green,
        yellow_ratio=yellow,
        brown_ratio=brown,
        dark_spot_ratio=dark,
        hue_histogram=(1.0,),
    )


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (_fs(green=0.8, yellow=0.2), Ripeness.INMADURO),
        (_fs(yellow=0.9, brown=0.1), Ripeness.COMESTIBLE),
        (_fs(yellow=0.3, brown=0.7), Ripeness.NO_COMESTIBLE),
        (_fs(yellow=0.6, dark=0.3), Ripeness.NO_COMESTIBLE),
        # Prioridad: "No comestible" gana aunque también haya mucho verde
        (_fs(green=0.5, brown=0.5), Ripeness.NO_COMESTIBLE),
    ],
)
def test_rules(features: FeatureSet, expected: Ripeness, thresholds: Thresholds) -> None:
    v = classify(features, thresholds.classifier)
    assert v.label is expected
    assert 0.0 <= v.confidence <= 1.0
    assert v.reasons
    assert v.features is features


def test_confidence_zero_at_threshold() -> None:
    assert margin_confidence(0.4, 0.4) == 0.0


def test_confidence_saturates() -> None:
    assert margin_confidence(1.0, 0.1) <= 1.0
