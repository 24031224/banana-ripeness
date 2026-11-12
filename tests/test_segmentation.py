from __future__ import annotations

import cv2
import numpy as np
import pytest

from banana.common import BGRImage, SegmentationError
from banana.config import Thresholds
from banana.segmentation import clean_mask, color_mask, largest_component, segment

pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="Esqueleto: pendiente")


@pytest.mark.parametrize("fixture_name", ["green_banana", "yellow_banana", "brown_banana"])
def test_segment_finds_banana_of_any_ripeness(
    fixture_name: str, request: pytest.FixtureRequest, thresholds: Thresholds
) -> None:
    img: BGRImage = request.getfixturevalue(fixture_name)
    seg = segment(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), thresholds.segmentation)
    assert set(np.unique(seg.mask)) <= {0, 255}
    assert 0.1 < seg.area_ratio < 0.6
    assert seg.area_px == int((seg.mask == 255).sum())


def test_segment_empty_raises(empty_image: BGRImage, thresholds: Thresholds) -> None:
    with pytest.raises(SegmentationError):
        segment(cv2.cvtColor(empty_image, cv2.COLOR_BGR2HSV), thresholds.segmentation)


def test_color_mask_is_binary(yellow_banana: BGRImage, thresholds: Thresholds) -> None:
    hsv = cv2.cvtColor(yellow_banana, cv2.COLOR_BGR2HSV)
    m = color_mask(hsv, thresholds.segmentation.banana_hsv)
    assert m.shape == hsv.shape[:2]
    assert set(np.unique(m)) <= {0, 255}


def test_clean_mask_removes_isolated_pixel() -> None:
    m = np.zeros((50, 50), dtype=np.uint8)
    m[25, 25] = 255
    assert clean_mask(m, 3, 1).sum() == 0


def test_largest_component_keeps_biggest() -> None:
    m = np.zeros((50, 50), dtype=np.uint8)
    m[5:10, 5:10] = 255  # pequeño
    m[20:45, 20:45] = 255  # grande
    out = largest_component(m)
    assert out[30, 30] == 255
    assert out[7, 7] == 0
