"""Segmentación: separa el plátano del fondo usando color HSV y morfología.

Entrada: imagen HSV (H en 0-179). Salida: :class:`banana.common.Segmentation`.
El rango ``banana_hsv`` debe cubrir plátanos verdes, amarillos Y cafés; si un
plátano muy maduro queda fuera de la máscara, el clasificador nunca lo verá café.
"""

from __future__ import annotations

import cv2
import numpy as np

from banana.common import HSVImage, HSVRange, Mask, Segmentation, SegmentationError
from banana.config import SegmentationConfig


def color_mask(hsv: HSVImage, hsv_range: HSVRange) -> Mask:
    """Genera una máscara con los píxeles dentro de un rango HSV (``cv2.inRange``).

    Args:
        hsv: Imagen HSV de OpenCV (H en 0-179).
        hsv_range: Rango inclusivo a conservar.

    Returns:
        Máscara binaria 0/255 de tamaño HxW.

    Examples:
        >>> m = color_mask(hsv, cfg.banana_hsv)
        >>> set(np.unique(m)) <= {0, 255}
        True
    """
    lower = np.array(hsv_range.lower, dtype=np.uint8)  # H en 0-179
    upper = np.array(hsv_range.upper, dtype=np.uint8)  # H en 0-179
    mask: Mask = cv2.inRange(hsv, lower, upper)
    return mask


def clean_mask(mask: Mask, kernel_size: int, iterations: int) -> Mask:
    """Elimina ruido y rellena huecos con apertura seguida de cierre morfológico.

    Args:
        mask: Máscara binaria 0/255.
        kernel_size: Tamaño del elemento estructurante elíptico (impar).
        iterations: Repeticiones de cada operación (>= 0; 0 = sin cambios).

    Returns:
        Máscara binaria 0/255 limpia, mismo tamaño.

    Raises:
        ValueError: Si ``kernel_size`` es par o < 1, o ``iterations`` < 0.

    Examples:
        >>> limpia = clean_mask(m, 7, 2)
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size debe ser impar y >= 1, se recibió {kernel_size}")
    if iterations < 0:
        raise ValueError(f"iterations debe ser >= 0, se recibió {iterations}")
    if iterations == 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed: Mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def largest_component(mask: Mask) -> Mask:
    """Conserva solo el contorno externo de mayor área, con su interior relleno.

    Args:
        mask: Máscara binaria 0/255.

    Returns:
        Máscara 0/255 con un único componente; toda en ceros si la entrada está vacía.

    Examples:
        >>> unico = largest_component(limpia)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result: Mask = np.zeros_like(mask)
    if not contours:
        return result
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(result, [largest], -1, 255, cv2.FILLED)
    return result


def segment(hsv: HSVImage, cfg: SegmentationConfig) -> Segmentation:
    """Ejecuta la etapa completa: rango de color → limpieza → componente mayor → geometría.

    Args:
        hsv: Imagen HSV preprocesada (H en 0-179).
        cfg: Sección ``segmentation`` de los umbrales.

    Returns:
        ``Segmentation`` con máscara, caja envolvente, área en píxeles y proporción.

    Raises:
        SegmentationError: Si el área resultante es menor que ``cfg.min_area_ratio``.

    Examples:
        >>> seg = segment(pre.hsv, thresholds.segmentation)
        >>> 0.0 < seg.area_ratio <= 1.0
        True
    """
    raw = color_mask(hsv, cfg.banana_hsv)
    cleaned = clean_mask(raw, cfg.morph_kernel, cfg.morph_iterations)
    mask = largest_component(cleaned)

    area_px = int(cv2.countNonZero(mask))
    area_ratio = area_px / float(mask.shape[0] * mask.shape[1])
    if area_px == 0:
        raise SegmentationError("No se detectó un plátano en la imagen.")
    if area_ratio < cfg.min_area_ratio:
        raise SegmentationError(
            f"El plátano ocupa solo {area_ratio:.1%} de la imagen "
            f"(mínimo {cfg.min_area_ratio:.1%})."
        )

    x, y, w, h = cv2.boundingRect(mask)
    return Segmentation(mask=mask, bbox=(x, y, w, h), area_px=area_px, area_ratio=area_ratio)
