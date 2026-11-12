from __future__ import annotations

import cv2
import numpy as np
import pytest

from banana.common import BGRImage, Segmentation
from banana.config import Thresholds
from banana.features import (
    color_ratio,
    dark_spot_ratio,
    extract_features,
    hsv_stats,
    hue_histogram,
)

pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="Esqueleto: pendiente")


def _seg_from_color(img: BGRImage) -> tuple[np.ndarray, Segmentation]:
    """Segmentación "perfecta" para aislar las pruebas de features de la segmentación."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.where(hsv[..., 1] > 0, 255, 0).astype(np.uint8)  # el fondo blanco tiene S=0
    area = int((mask == 255).sum())
    seg = Segmentation(mask=mask, bbox=(0, 0, 1, 1), area_px=area, area_ratio=area / mask.size)
    return hsv, seg


def test_green_banana_is_mostly_green(green_banana: BGRImage, thresholds: Thresholds) -> None:
    hsv, seg = _seg_from_color(green_banana)
    assert color_ratio(hsv, seg.mask, thresholds.features.green) > 0.9


def test_histogram_normalized(yellow_banana: BGRImage) -> None:
    hsv, seg = _seg_from_color(yellow_banana)
    h = hue_histogram(hsv, seg.mask, 18)
    assert len(h) == 18
    assert sum(h) == pytest.approx(1.0)


def test_dark_spots_zero_on_bright_banana(yellow_banana: BGRImage) -> None:
    hsv, seg = _seg_from_color(yellow_banana)
    assert dark_spot_ratio(hsv, seg.mask, 60) == 0.0


def test_hsv_stats_hue_in_opencv_scale(yellow_banana: BGRImage) -> None:
    hsv, seg = _seg_from_color(yellow_banana)
    h_mean, _, _, _ = hsv_stats(hsv, seg.mask)
    assert 20 <= h_mean <= 35  # amarillo en escala 0-179


def test_extract_features_ratios_in_unit_interval(
    brown_banana: BGRImage, thresholds: Thresholds
) -> None:
    hsv, seg = _seg_from_color(brown_banana)
    fs = extract_features(hsv, seg, thresholds.features)
    for ratio in (fs.green_ratio, fs.yellow_ratio, fs.brown_ratio, fs.dark_spot_ratio):
        assert 0.0 <= ratio <= 1.0
    assert fs.brown_ratio > 0.9
