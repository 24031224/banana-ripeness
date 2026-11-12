from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from banana.common import BGRImage
from banana.config import Thresholds
from banana.preprocess import (
    apply_white_balance,
    denoise,
    preprocess,
    resize_to_width,
    to_hsv,
)


def test_resize_rejects_non_positive_width(yellow_banana: BGRImage) -> None:
    with pytest.raises(ValueError):
        resize_to_width(yellow_banana, 0)


def test_denoise_kernel_one_returns_unchanged_copy(yellow_banana: BGRImage) -> None:
    out = denoise(yellow_banana, 1)
    assert out is not yellow_banana
    assert np.array_equal(out, yellow_banana)


def test_denoise_smooths_noise() -> None:
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
    assert denoise(noisy, 5).std() < noisy.std()


def test_white_balance_gray_world_equalizes_channel_means() -> None:
    tinted = np.empty((20, 20, 3), dtype=np.uint8)
    tinted[:] = (60, 100, 200)  # BGR con dominante roja
    out = apply_white_balance(tinted, "gray_world")
    means = out.reshape(-1, 3).mean(axis=0)
    assert means.max() - means.min() <= 2.0


def test_white_balance_black_image_is_unchanged() -> None:
    black = np.zeros((10, 10, 3), dtype=np.uint8)
    assert np.array_equal(apply_white_balance(black, "gray_world"), black)


def test_white_balance_rejects_unknown_method(yellow_banana: BGRImage) -> None:
    with pytest.raises(ValueError):
        apply_white_balance(yellow_banana, "magic")  # type: ignore[arg-type]


def test_preprocess_reports_scale(yellow_banana: BGRImage, thresholds: Thresholds) -> None:
    pre = preprocess(yellow_banana, thresholds.preprocess)
    expected = thresholds.preprocess.target_width / yellow_banana.shape[1]
    assert pre.scale == pytest.approx(expected)


def test_preprocess_gray_world_pipeline(yellow_banana: BGRImage, thresholds: Thresholds) -> None:
    cfg = replace(thresholds.preprocess, target_width=160, white_balance="gray_world")
    pre = preprocess(yellow_banana, cfg)
    assert pre.bgr.dtype == np.uint8
    assert pre.bgr.shape[1] == 160
    assert int(pre.hsv[..., 0].max()) <= 179  # OpenCV: H en 0-179


def test_resize_keeps_aspect_ratio(yellow_banana: BGRImage) -> None:
    out, scale = resize_to_width(yellow_banana, 160)
    assert out.shape[1] == 160
    assert out.shape[0] == round(yellow_banana.shape[0] * scale)


def test_denoise_rejects_even_kernel(yellow_banana: BGRImage) -> None:
    with pytest.raises(ValueError):
        denoise(yellow_banana, 4)


def test_white_balance_none_is_identity(yellow_banana: BGRImage) -> None:
    assert (apply_white_balance(yellow_banana, "none") == yellow_banana).all()


def test_hsv_hue_in_opencv_range(yellow_banana: BGRImage) -> None:
    hsv = to_hsv(yellow_banana)
    assert int(hsv[..., 0].max()) <= 179  # OpenCV: H en 0-179


def test_preprocess_shapes(yellow_banana: BGRImage, thresholds: Thresholds) -> None:
    pre = preprocess(yellow_banana, thresholds.preprocess)
    assert pre.bgr.shape == pre.hsv.shape
    assert pre.bgr.shape[1] == thresholds.preprocess.target_width
