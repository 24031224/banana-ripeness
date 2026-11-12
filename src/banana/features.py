"""Extracción de características de color, calculadas SOLO dentro de la máscara.

Entrada: imagen HSV + :class:`banana.common.Segmentation`.
Salida: :class:`banana.common.FeatureSet`.

Recordatorio: H de OpenCV va de 0 a 179. El tono es circular (179 está junto a 0);
para plátanos (verde ~35-85, amarillo ~20-35, café ~5-20) la media lineal es
aceptable, pero documenta cualquier caso donde no lo sea.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from banana.common import FeatureSet, HSVImage, HSVRange, Mask, Segmentation
from banana.config import FeaturesConfig
from banana.segmentation import color_mask

_HUE_RANGE_END = 180
"""Límite superior exclusivo de H en OpenCV (H va de 0 a 179)."""


def hue_histogram(hsv: HSVImage, mask: Mask, bins: int) -> tuple[float, ...]:
    """Calcula el histograma normalizado del canal H dentro de la máscara.

    Args:
        hsv: Imagen HSV de OpenCV.
        mask: Máscara 0/255 del plátano.
        bins: Número de cubetas sobre el rango 0-179 (> 0).

    Returns:
        Tupla de ``bins`` valores que suman 1.0 (todo ceros si la máscara está vacía).

    Raises:
        ValueError: Si ``bins`` <= 0.

    Examples:
        >>> h = hue_histogram(hsv, seg.mask, 18)
        >>> round(sum(h), 6)
        1.0
    """
    if bins <= 0:
        raise ValueError(f"bins debe ser > 0, se recibió {bins}")
    hist = cv2.calcHist([hsv], [0], mask, [bins], [0, _HUE_RANGE_END]).flatten()  # H en 0-179
    total = float(hist.sum())
    if total == 0.0:
        return (0.0,) * bins
    return tuple(float(v) for v in hist / total)


def color_ratio(hsv: HSVImage, mask: Mask, hsv_range: HSVRange) -> float:
    """Fracción de píxeles del plátano que caen dentro de un rango HSV.

    Args:
        hsv: Imagen HSV de OpenCV (H en 0-179).
        mask: Máscara 0/255 del plátano.
        hsv_range: Rango de color a contar (p. ej. ``cfg.green``).

    Returns:
        Proporción en [0, 1]; 0.0 si la máscara está vacía.

    Examples:
        >>> verde = color_ratio(hsv, seg.mask, cfg.green)
    """
    banana_pixels = int(cv2.countNonZero(mask))
    if banana_pixels == 0:
        return 0.0
    inside = cv2.bitwise_and(color_mask(hsv, hsv_range), mask)
    return float(cv2.countNonZero(inside)) / banana_pixels


def other_ratio(hsv: HSVImage, mask: Mask, ranges: Sequence[HSVRange]) -> float:
    """Fracción de píxeles del plátano que no caen en ninguno de los rangos dados.

    Los rangos pueden solaparse (p. ej. amarillo y café comparten H=20), por eso se
    cuenta la unión de las máscaras y no se resta la suma de proporciones.

    Args:
        hsv: Imagen HSV de OpenCV (H en 0-179).
        mask: Máscara 0/255 del plátano.
        ranges: Rangos HSV considerados "conocidos" (verde, amarillo, café).

    Returns:
        Proporción en [0, 1]; 0.0 si la máscara está vacía.

    Examples:
        >>> otros = other_ratio(hsv, seg.mask, (cfg.green, cfg.yellow, cfg.brown))
    """
    banana_pixels = int(cv2.countNonZero(mask))
    if banana_pixels == 0:
        return 0.0
    known = np.zeros(mask.shape, dtype=np.uint8)
    for hsv_range in ranges:
        known = cv2.bitwise_or(known, color_mask(hsv, hsv_range))
    covered = cv2.countNonZero(cv2.bitwise_and(known, mask))
    return float(banana_pixels - covered) / banana_pixels


def dark_spot_ratio(hsv: HSVImage, mask: Mask, value_max: int) -> float:
    """Fracción de píxeles del plátano con brillo (V) menor que ``value_max``.

    Args:
        hsv: Imagen HSV de OpenCV.
        mask: Máscara 0/255 del plátano.
        value_max: Umbral de V (0-255) por debajo del cual un píxel es "mancha".

    Returns:
        Proporción en [0, 1]; 0.0 si la máscara está vacía.

    Examples:
        >>> manchas = dark_spot_ratio(hsv, seg.mask, cfg.dark_spot_value_max)
    """
    banana_pixels = int(cv2.countNonZero(mask))
    if banana_pixels == 0:
        return 0.0
    dark = (hsv[:, :, 2] < value_max) & (mask == 255)  # V en 0-255
    return float(np.count_nonzero(dark)) / banana_pixels


def hsv_stats(hsv: HSVImage, mask: Mask) -> tuple[float, float, float, float]:
    """Estadísticos básicos de los canales HSV dentro de la máscara.

    Args:
        hsv: Imagen HSV de OpenCV.
        mask: Máscara 0/255 del plátano.

    Returns:
        Tupla ``(hue_mean, hue_std, saturation_mean, value_mean)``; H en escala 0-179.

    Raises:
        ValueError: Si la máscara está vacía.

    Examples:
        >>> h_mean, h_std, s_mean, v_mean = hsv_stats(hsv, seg.mask)
    """
    inside = hsv[mask == 255]
    if inside.shape[0] == 0:
        raise ValueError("La máscara está vacía: no hay píxeles de plátano.")
    hue = inside[:, 0].astype(np.float64)  # H en 0-179
    return (
        float(hue.mean()),
        float(hue.std()),
        float(inside[:, 1].mean()),
        float(inside[:, 2].mean()),
    )


def extract_features(hsv: HSVImage, segmentation: Segmentation, cfg: FeaturesConfig) -> FeatureSet:
    """Ejecuta la etapa completa y arma el ``FeatureSet``.

    Args:
        hsv: Imagen HSV preprocesada (H en 0-179).
        segmentation: Resultado de :func:`banana.segmentation.segment`.
        cfg: Sección ``features`` de los umbrales.

    Returns:
        ``FeatureSet`` con estadísticos, proporciones de color e histograma.

    Raises:
        ValueError: Si la máscara de ``segmentation`` está vacía.

    Examples:
        >>> fs = extract_features(pre.hsv, seg, thresholds.features)
        >>> 0.0 <= fs.green_ratio <= 1.0
        True
    """
    mask = segmentation.mask
    hue_mean, hue_std, saturation_mean, value_mean = hsv_stats(hsv, mask)
    return FeatureSet(
        hue_mean=hue_mean,
        hue_std=hue_std,
        saturation_mean=saturation_mean,
        value_mean=value_mean,
        green_ratio=color_ratio(hsv, mask, cfg.green),
        yellow_ratio=color_ratio(hsv, mask, cfg.yellow),
        brown_ratio=color_ratio(hsv, mask, cfg.brown),
        dark_spot_ratio=dark_spot_ratio(hsv, mask, cfg.dark_spot_value_max),
        hue_histogram=hue_histogram(hsv, mask, cfg.hue_bins),
        other_ratio=other_ratio(hsv, mask, (cfg.green, cfg.yellow, cfg.brown)),
    )
