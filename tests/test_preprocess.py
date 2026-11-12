from __future__ import annotations

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

pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="Esqueleto: pendiente")


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
