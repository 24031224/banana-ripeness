from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from banana.common import BGRImage, SegmentationError
from banana.config import Thresholds
from banana.segmentation import clean_mask, color_mask, largest_component, segment


def test_segment_bbox_matches_mask(yellow_banana: BGRImage, thresholds: Thresholds) -> None:
    seg = segment(cv2.cvtColor(yellow_banana, cv2.COLOR_BGR2HSV), thresholds.segmentation)
    x, y, w, h = seg.bbox
    assert w > h > 0  # la elipse sintética es alargada
    outside = seg.mask.copy()
    outside[y : y + h, x : x + w] = 0
    assert outside.sum() == 0  # nada de máscara fuera de la caja
    assert seg.area_ratio == pytest.approx(seg.area_px / seg.mask.size)


def test_segment_too_small_raises(yellow_banana: BGRImage, thresholds: Thresholds) -> None:
    cfg = replace(thresholds.segmentation, min_area_ratio=0.99)
    with pytest.raises(SegmentationError):
        segment(cv2.cvtColor(yellow_banana, cv2.COLOR_BGR2HSV), cfg)


def test_clean_mask_zero_iterations_is_copy() -> None:
    m = np.zeros((20, 20), dtype=np.uint8)
    m[10, 10] = 255
    out = clean_mask(m, 3, 0)
    assert out is not m
    assert np.array_equal(out, m)


@pytest.mark.parametrize(("kernel", "iterations"), [(4, 1), (0, 1), (3, -1)])
def test_clean_mask_rejects_invalid_params(kernel: int, iterations: int) -> None:
    with pytest.raises(ValueError):
        clean_mask(np.zeros((10, 10), dtype=np.uint8), kernel, iterations)


def test_largest_component_empty_mask_returns_zeros() -> None:
    out = largest_component(np.zeros((10, 10), dtype=np.uint8))
    assert out.sum() == 0


def test_largest_component_fills_holes() -> None:
    m = np.zeros((50, 50), dtype=np.uint8)
    m[10:40, 10:40] = 255
    m[20:30, 20:30] = 0  # hueco interior
    assert largest_component(m)[25, 25] == 255


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
