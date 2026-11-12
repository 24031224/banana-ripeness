"""Pruebas de integración de extremo a extremo con imágenes sintéticas."""

from __future__ import annotations

import pytest

from banana.common import BGRImage, Ripeness, SegmentationError
from banana.config import Thresholds
from banana.pipeline import BananaPipeline

pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="Esqueleto: pendiente")


@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [
        ("green_banana", Ripeness.INMADURO),
        ("yellow_banana", Ripeness.COMESTIBLE),
        ("brown_banana", Ripeness.NO_COMESTIBLE),
    ],
)
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


def test_no_banana_raises(empty_image: BGRImage, thresholds: Thresholds) -> None:
    with pytest.raises(SegmentationError):
        BananaPipeline(thresholds).run(empty_image)


def test_from_yaml_default() -> None:
    assert isinstance(BananaPipeline.from_yaml(), BananaPipeline)
