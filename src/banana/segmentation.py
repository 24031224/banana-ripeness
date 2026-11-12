"""Segmentación: separa el plátano del fondo usando color HSV y morfología.

Entrada: imagen HSV (H en 0-179). Salida: :class:`banana.common.Segmentation`.
El rango ``banana_hsv`` debe cubrir plátanos verdes, amarillos Y cafés; si un
plátano muy maduro queda fuera de la máscara, el clasificador nunca lo verá café.
"""

from __future__ import annotations

from banana.common import HSVImage, HSVRange, Mask, Segmentation
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
    raise NotImplementedError


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
    raise NotImplementedError


def largest_component(mask: Mask) -> Mask:
    """Conserva solo el contorno externo de mayor área, con su interior relleno.

    Args:
        mask: Máscara binaria 0/255.

    Returns:
        Máscara 0/255 con un único componente; toda en ceros si la entrada está vacía.

    Examples:
        >>> unico = largest_component(limpia)
    """
    raise NotImplementedError


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
    raise NotImplementedError
