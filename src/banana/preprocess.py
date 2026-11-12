"""Preprocesamiento: normaliza tamaño, ruido e iluminación y convierte a HSV.

Entrada: imagen BGR cruda. Salida: :class:`banana.common.PreprocessedImage`.
Recordatorio: ``cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`` produce H en 0-179, no 0-360.
"""

from __future__ import annotations

import cv2
import numpy as np

from banana.common import BGRImage, HSVImage, PreprocessedImage
from banana.config import PreprocessConfig, WhiteBalanceMethod

_EPSILON = 1e-6
"""Evita dividir entre cero al escalar canales de media nula (no es un umbral de color)."""


def resize_to_width(image: BGRImage, target_width: int) -> tuple[BGRImage, float]:
    """Redimensiona conservando la proporción para que el ancho sea ``target_width``.

    Args:
        image: Imagen BGR de entrada.
        target_width: Ancho deseado en píxeles (> 0).

    Returns:
        Tupla ``(imagen_redimensionada, escala)`` con ``escala = target_width / ancho_original``.

    Raises:
        ValueError: Si ``target_width`` <= 0.

    Examples:
        >>> small, scale = resize_to_width(img, 640)
        >>> small.shape[1]
        640
    """
    if target_width <= 0:
        raise ValueError(f"target_width debe ser > 0, se recibió {target_width}")
    height, width = image.shape[:2]
    scale = target_width / width
    new_height = max(1, int(height * scale))
    resized: BGRImage = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def denoise(image: BGRImage, kernel_size: int) -> BGRImage:
    """Aplica suavizado gaussiano para reducir ruido de sensor.

    Args:
        image: Imagen BGR.
        kernel_size: Tamaño del kernel; impar y >= 1. Con 1 se devuelve una copia sin cambios.

    Returns:
        Imagen BGR suavizada, mismo tamaño.

    Raises:
        ValueError: Si ``kernel_size`` es par o menor que 1.

    Examples:
        >>> smooth = denoise(img, 5)
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size debe ser impar y >= 1, se recibió {kernel_size}")
    if kernel_size == 1:
        return image.copy()
    blurred: BGRImage = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


def apply_white_balance(image: BGRImage, method: WhiteBalanceMethod) -> BGRImage:
    """Corrige el tono de la iluminación para que el color dependa del plátano, no de la luz.

    ``"gray_world"`` escala cada canal para que su media iguale la media global;
    ``"none"`` devuelve una copia sin cambios.

    Args:
        image: Imagen BGR.
        method: Método de balance de blancos.

    Returns:
        Imagen BGR corregida, ``uint8``, mismo tamaño.

    Raises:
        ValueError: Si ``method`` no es un método soportado.

    Examples:
        >>> balanced = apply_white_balance(img, "gray_world")
    """
    if method == "none":
        return image.copy()
    if method != "gray_world":
        raise ValueError(f"Método de balance de blancos no soportado: {method!r}")

    channel_means = image.mean(axis=(0, 1))
    global_mean = float(channel_means.mean())
    if global_mean <= 0:
        return image.copy()  # imagen completamente negra: no hay nada que corregir
    gains = global_mean / (channel_means + _EPSILON)
    balanced = np.clip(image * gains[np.newaxis, np.newaxis, :], 0, 255).astype(np.uint8)
    return balanced


def to_hsv(image: BGRImage) -> HSVImage:
    """Convierte BGR a HSV de OpenCV.

    Ojo: OpenCV usa H en 0-179 (grados / 2), S y V en 0-255.

    Args:
        image: Imagen BGR ``uint8``.

    Returns:
        Imagen HSV ``uint8`` del mismo tamaño.

    Examples:
        >>> hsv = to_hsv(img)
        >>> int(hsv[..., 0].max()) <= 179
        True
    """
    hsv: HSVImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # H en 0-179
    return hsv


def preprocess(image: BGRImage, cfg: PreprocessConfig) -> PreprocessedImage:
    """Ejecuta la etapa completa: redimensionar → suavizar → balance de blancos → HSV.

    Args:
        image: Imagen BGR cruda de :mod:`banana.acquisition`.
        cfg: Sección ``preprocess`` de los umbrales.

    Returns:
        ``PreprocessedImage`` con la versión BGR, la HSV y la escala aplicada.

    Examples:
        >>> pre = preprocess(img, thresholds.preprocess)
        >>> pre.hsv.shape == pre.bgr.shape
        True
    """
    resized, scale = resize_to_width(image, cfg.target_width)
    smooth = denoise(resized, cfg.blur_kernel)
    balanced = apply_white_balance(smooth, cfg.white_balance)
    return PreprocessedImage(bgr=balanced, hsv=to_hsv(balanced), scale=scale)
