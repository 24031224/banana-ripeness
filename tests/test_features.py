from __future__ import annotations

import cv2
import numpy as np
import pytest

from banana.common import BGRImage, HSVRange, Segmentation
from banana.config import Thresholds
from banana.features import (
    color_ratio,
    dark_spot_ratio,
    extract_features,
    hsv_stats,
    hue_histogram,
    other_ratio,
)


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


def test_empty_mask_gives_zero_ratios_and_histogram(
    yellow_banana: BGRImage, thresholds: Thresholds
) -> None:
    hsv = cv2.cvtColor(yellow_banana, cv2.COLOR_BGR2HSV)
    empty = np.zeros(hsv.shape[:2], dtype=np.uint8)
    assert color_ratio(hsv, empty, thresholds.features.yellow) == 0.0
    assert dark_spot_ratio(hsv, empty, 60) == 0.0
    assert hue_histogram(hsv, empty, 18) == (0.0,) * 18


def test_hsv_stats_empty_mask_raises(yellow_banana: BGRImage) -> None:
    hsv = cv2.cvtColor(yellow_banana, cv2.COLOR_BGR2HSV)
    with pytest.raises(ValueError):
        hsv_stats(hsv, np.zeros(hsv.shape[:2], dtype=np.uint8))


def test_histogram_rejects_non_positive_bins(yellow_banana: BGRImage) -> None:
    hsv, seg = _seg_from_color(yellow_banana)
    with pytest.raises(ValueError):
        hue_histogram(hsv, seg.mask, 0)


def test_histogram_peak_in_yellow_bin(yellow_banana: BGRImage) -> None:
    hsv, seg = _seg_from_color(yellow_banana)
    h = hue_histogram(hsv, seg.mask, 18)  # cubetas de 10 sobre H 0-179
    assert int(np.argmax(h)) == 2  # H≈27 cae en la cubeta 20-29


def test_dark_spots_detected_on_dark_pixels() -> None:
    hsv = np.full((10, 10, 3), (12, 180, 200), dtype=np.uint8)
    hsv[:5] = (12, 180, 20)  # mitad superior oscura (V=20)
    mask = np.full((10, 10), 255, dtype=np.uint8)
    assert dark_spot_ratio(hsv, mask, 60) == pytest.approx(0.5)


def test_color_ratio_ignores_pixels_outside_mask() -> None:
    hsv = np.full((10, 10, 3), (55, 200, 170), dtype=np.uint8)  # todo verde (H 0-179)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5] = 255
    green = HSVRange(lower=(36, 40, 40), upper=(85, 255, 255))
    assert color_ratio(hsv, mask, green) == pytest.approx(1.0)


def test_extract_features_fields_are_consistent(
    green_banana: BGRImage, thresholds: Thresholds
) -> None:
    hsv, seg = _seg_from_color(green_banana)
    fs = extract_features(hsv, seg, thresholds.features)
    assert 36 <= fs.hue_mean <= 85
    assert len(fs.hue_histogram) == thresholds.features.hue_bins
    assert sum(fs.hue_histogram) == pytest.approx(1.0)
    assert fs.green_ratio > 0.9


def test_other_ratio_counts_union_without_double_counting() -> None:
    green = HSVRange(lower=(36, 40, 40), upper=(85, 255, 255))
    yellow = HSVRange(lower=(20, 60, 100), upper=(35, 255, 255))
    hsv = np.zeros((10, 10, 3), dtype=np.uint8)
    hsv[:] = (55, 200, 170)  # verde (H en 0-179)
    hsv[:3] = (27, 200, 200)  # amarillo: 30 %
    hsv[3:5] = (120, 200, 200)  # azul, fuera de todos los rangos: 20 %
    mask = np.full((10, 10), 255, dtype=np.uint8)
    assert other_ratio(hsv, mask, (green, yellow)) == pytest.approx(0.2)
    # rangos solapados (el mismo dos veces) no cambian el resultado
    assert other_ratio(hsv, mask, (green, yellow, yellow)) == pytest.approx(0.2)


def test_other_ratio_empty_mask_is_zero(yellow_banana: BGRImage, thresholds: Thresholds) -> None:
    hsv = cv2.cvtColor(yellow_banana, cv2.COLOR_BGR2HSV)
    empty = np.zeros(hsv.shape[:2], dtype=np.uint8)
    assert other_ratio(hsv, empty, (thresholds.features.green,)) == 0.0


def test_extract_features_other_ratio(yellow_banana: BGRImage, thresholds: Thresholds) -> None:
    hsv, seg = _seg_from_color(yellow_banana)
    fs = extract_features(hsv, seg, thresholds.features)
    assert 0.0 <= fs.other_ratio <= 1.0
    assert fs.other_ratio < 0.1  # casi todo es amarillo


def test_extract_features_ratios_in_unit_interval(
    brown_banana: BGRImage, thresholds: Thresholds
) -> None:
    hsv, seg = _seg_from_color(brown_banana)
    fs = extract_features(hsv, seg, thresholds.features)
    for ratio in (fs.green_ratio, fs.yellow_ratio, fs.brown_ratio, fs.dark_spot_ratio):
        assert 0.0 <= ratio <= 1.0
    assert fs.brown_ratio > 0.9
