"""Pruebas de integración de extremo a extremo con imágenes sintéticas."""

from __future__ import annotations

import numpy as np
import pytest

from banana.common import BGRImage, ImageLoadError, Ripeness, SegmentationError
from banana.config import Thresholds
from banana.pipeline import BananaPipeline, analyze, overlay_mask

# Las etapas previas (config, preprocess, segmentation, features) pueden estar aún sin
# implementar; estos tests pasan a ser obligatorios cuando lo estén.
pending_stages = pytest.mark.xfail(raises=NotImplementedError, reason="Etapas previas pendientes")

_CASES = [
    ("green_banana", Ripeness.INMADURO),
    ("yellow_banana", Ripeness.COMESTIBLE),
    ("brown_banana", Ripeness.NO_COMESTIBLE),
]


@pending_stages
@pytest.mark.parametrize(("fixture_name", "expected"), _CASES)
def test_end_to_end(
    fixture_name: str,
    expected: Ripeness,
    request: pytest.FixtureRequest,
    thresholds: Thresholds,
) -> None:
    img: BGRImage = request.getfixturevalue(fixture_name)
    result = BananaPipeline(thresholds).run(img)
    assert result.verdict.label is expected
    assert result.segmentation.mask.shape == result.preprocessed.hsv.shape[:2]


@pending_stages
def test_no_banana_raises(empty_image: BGRImage, thresholds: Thresholds) -> None:
    with pytest.raises(SegmentationError):
        BananaPipeline(thresholds).run(empty_image)


@pending_stages
def test_from_yaml_default() -> None:
    assert isinstance(BananaPipeline.from_yaml(), BananaPipeline)


@pending_stages
@pytest.mark.parametrize(("fixture_name", "expected"), _CASES)
def test_analyze_from_array(
    fixture_name: str,
    expected: Ripeness,
    request: pytest.FixtureRequest,
    thresholds: Thresholds,
) -> None:
    img: BGRImage = request.getfixturevalue(fixture_name)
    verdict, features, overlay = analyze(img, BananaPipeline(thresholds))
    assert verdict.label is expected
    assert features is verdict.features
    assert overlay.dtype == np.uint8
    assert overlay.ndim == 3


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((10, 10, 4), dtype=np.uint8),
    ],
)
def test_analyze_rejects_bad_array(bad: BGRImage) -> None:
    with pytest.raises(ImageLoadError):
        analyze(bad)


def test_overlay_mask_tints_only_masked_area() -> None:
    img = np.full((40, 40, 3), 200, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    out = overlay_mask(img, mask)
    assert out.shape == img.shape
    assert not np.array_equal(out[20, 20], img[20, 20])
    assert np.array_equal(out[2, 2], img[2, 2])
    assert np.all(img == 200)  # la entrada no se modifica


def test_overlay_mask_size_mismatch() -> None:
    with pytest.raises(ValueError):
        overlay_mask(np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((5, 5), dtype=np.uint8))
