"""Extracción de características de color, calculadas SOLO dentro de la máscara.

Entrada: imagen HSV + :class:`banana.common.Segmentation`.
Salida: :class:`banana.common.FeatureSet`.

Recordatorio: H de OpenCV va de 0 a 179. El tono es circular (179 está junto a 0);
para plátanos (verde ~35-85, amarillo ~20-35, café ~5-20) la media lineal es
aceptable, pero documenta cualquier caso donde no lo sea.
"""

from __future__ import annotations

from banana.common import FeatureSet, HSVImage, HSVRange, Mask, Segmentation
from banana.config import FeaturesConfig


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def extract_features(hsv: HSVImage, segmentation: Segmentation, cfg: FeaturesConfig) -> FeatureSet:
    """Ejecuta la etapa completa y arma el ``FeatureSet``.

    Args:
        hsv: Imagen HSV preprocesada (H en 0-179).
        segmentation: Resultado de :func:`banana.segmentation.segment`.
        cfg: Sección ``features`` de los umbrales.

    Returns:
        ``FeatureSet`` con estadísticos, proporciones de color e histograma.

    Examples:
        >>> fs = extract_features(pre.hsv, seg, thresholds.features)
        >>> 0.0 <= fs.green_ratio <= 1.0
        True
    """
    raise NotImplementedError
