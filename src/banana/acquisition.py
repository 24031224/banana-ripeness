"""Adquisición de imágenes: archivos, bytes (subidas web), cámara y datasets.

Todas las funciones devuelven imágenes **BGR uint8** (convención de OpenCV).
Ninguna función de este módulo modifica la imagen (eso es tarea de ``preprocess``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from banana.common import BGRImage, LabeledSample

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
"""Extensiones aceptadas al recorrer carpetas (no es un umbral, es formato de archivo)."""


def load_image(path: Path | str) -> BGRImage:
    """Lee una imagen desde disco en formato BGR.

    Args:
        path: Ruta al archivo de imagen.

    Returns:
        Imagen HxWx3 ``uint8`` en BGR.

    Raises:
        ImageLoadError: Si el archivo no existe o OpenCV no puede decodificarlo
            (``cv2.imread`` devuelve ``None`` en lugar de lanzar error).

    Examples:
        >>> img = load_image("data/samples/comestible_01.jpg")
        >>> img.shape[2]
        3
    """
    raise NotImplementedError


def load_from_bytes(data: bytes) -> BGRImage:
    """Decodifica una imagen codificada (JPG/PNG) desde bytes, p. ej. de Streamlit.

    Args:
        data: Contenido binario del archivo de imagen.

    Returns:
        Imagen HxWx3 ``uint8`` en BGR.

    Raises:
        ImageLoadError: Si los bytes están vacíos o no son una imagen válida.

    Examples:
        >>> img = load_from_bytes(uploaded_file.getvalue())
    """
    raise NotImplementedError


def capture_from_camera(device_index: int = 0) -> BGRImage:
    """Captura un solo cuadro de una cámara conectada.

    Args:
        device_index: Índice del dispositivo para ``cv2.VideoCapture``.

    Returns:
        Cuadro capturado en BGR.

    Raises:
        ImageLoadError: Si la cámara no abre o no entrega cuadro.

    Examples:
        >>> frame = capture_from_camera(0)
    """
    raise NotImplementedError


def iter_dataset(root: Path | str) -> Iterator[LabeledSample]:
    """Recorre un dataset organizado en subcarpetas por clase.

    Estructura esperada (ver ``banana.common.DATASET_FOLDERS``)::

        root/
          inmaduro/*.jpg
          comestible/*.jpg
          no_comestible/*.jpg

    Las subcarpetas desconocidas se ignoran; el orden es determinista (ordenado por ruta).

    Args:
        root: Carpeta raíz del dataset.

    Yields:
        Una ``LabeledSample`` por cada imagen con extensión en ``IMAGE_EXTENSIONS``.

    Raises:
        FileNotFoundError: Si ``root`` no existe.

    Examples:
        >>> muestras = list(iter_dataset("data/raw"))
        >>> muestras[0].label
        <Ripeness.COMESTIBLE: 'Comestible'>
    """
    raise NotImplementedError
