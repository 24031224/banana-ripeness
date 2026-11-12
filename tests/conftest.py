"""Fixtures compartidas.

Las configuraciones se construyen A MANO (no desde el YAML) para que cada módulo
pueda probarse en aislamiento aunque ``load_thresholds`` aún no exista. Son
valores de prueba, independientes de los de producción.

Las imágenes sintéticas se generan en HSV (H en 0-179, escala OpenCV) y se
convierten a BGR, así conocemos exactamente el color del "plátano".
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from banana.common import BGRImage, HSVRange
from banana.config import (
    ClassifierConfig,
    FeaturesConfig,
    PreprocessConfig,
    SegmentationConfig,
    Thresholds,
)

IMG_H, IMG_W = 240, 320

# Colores HSV de prueba (H en 0-179)
HSV_GREEN = (55, 200, 170)
HSV_YELLOW = (27, 200, 230)
HSV_BROWN = (12, 180, 90)
HSV_WHITE_BG = (0, 0, 255)


def make_banana_image(hsv_color: tuple[int, int, int]) -> BGRImage:
    """Fondo blanco con una elipse alargada del color HSV dado, devuelta en BGR."""
    hsv = np.full((IMG_H, IMG_W, 3), HSV_WHITE_BG, dtype=np.uint8)
    cv2.ellipse(hsv, (IMG_W // 2, IMG_H // 2), (120, 40), 15, 0, 360, hsv_color, thickness=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@pytest.fixture
def green_banana() -> BGRImage:
    return make_banana_image(HSV_GREEN)


@pytest.fixture
def yellow_banana() -> BGRImage:
    return make_banana_image(HSV_YELLOW)


@pytest.fixture
def brown_banana() -> BGRImage:
    return make_banana_image(HSV_BROWN)


@pytest.fixture
def empty_image() -> BGRImage:
    """Solo fondo blanco: no hay plátano que segmentar."""
    hsv = np.full((IMG_H, IMG_W, 3), HSV_WHITE_BG, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@pytest.fixture
def thresholds() -> Thresholds:
    return Thresholds(
        version=1,
        preprocess=PreprocessConfig(target_width=320, blur_kernel=3, white_balance="none"),
        segmentation=SegmentationConfig(
            banana_hsv=HSVRange(lower=(5, 40, 30), upper=(90, 255, 255)),
            morph_kernel=5,
            morph_iterations=1,
            min_area_ratio=0.02,
        ),
        features=FeaturesConfig(
            hue_bins=18,
            green=HSVRange(lower=(36, 40, 40), upper=(85, 255, 255)),
            yellow=HSVRange(lower=(20, 60, 100), upper=(35, 255, 255)),
            brown=HSVRange(lower=(5, 40, 30), upper=(20, 255, 150)),
            dark_spot_value_max=60,
        ),
        classifier=ClassifierConfig(
            no_comestible_brown_ratio_min=0.40,
            no_comestible_dark_ratio_min=0.25,
            inmaduro_green_ratio_min=0.35,
        ),
    )
