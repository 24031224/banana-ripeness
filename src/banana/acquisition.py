"""Adquisición de imágenes: archivos, bytes (subidas web), cámara y datasets.

Todas las funciones devuelven imágenes **BGR uint8** (convención de OpenCV).
Ninguna función de este módulo modifica la imagen (eso es tarea de ``preprocess``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from banana.common import DATASET_FOLDERS, BGRImage, ImageLoadError, LabeledSample

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
    image_path = Path(path)
    if not image_path.is_file():
        raise ImageLoadError(f"No existe el archivo de imagen: {image_path}")
    try:
        # np.fromfile + imdecode: cv2.imread falla con tildes/espacios en rutas de Windows.
        buf = np.fromfile(image_path, dtype=np.uint8)
    except OSError as exc:
        raise ImageLoadError(f"No se pudo leer {image_path}: {exc}") from exc
    return _decode(buf, str(image_path))


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
    if not data:
        raise ImageLoadError("Los bytes de la imagen están vacíos")
    return _decode(np.frombuffer(data, dtype=np.uint8), "<bytes>")


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
    cap = cv2.VideoCapture(device_index)
    try:
        if not cap.isOpened():
            raise ImageLoadError(f"No se pudo abrir la cámara {device_index}")
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise ImageLoadError(f"La cámara {device_index} no entregó ningún cuadro")
    return frame


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
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {root_path}")
    files = sorted(f for f in root_path.glob("*/*") if f.is_file())
    for file in files:
        label = DATASET_FOLDERS.get(file.parent.name)
        if label is not None and file.suffix.lower() in IMAGE_EXTENSIONS:
            yield LabeledSample(path=file, label=label)


def _decode(buf: npt.NDArray[np.uint8], origin: str) -> BGRImage:
    """Decodifica un buffer a BGR ``uint8`` o lanza ``ImageLoadError``.

    Args:
        buf: Bytes de la imagen codificada como arreglo 1D ``uint8``.
        origin: Texto que identifica el origen, para el mensaje de error.

    Returns:
        Imagen HxWx3 ``uint8`` en BGR.

    Raises:
        ImageLoadError: Si el buffer está vacío o no es una imagen decodificable.

    Examples:
        >>> _decode(np.frombuffer(png_bytes, dtype=np.uint8), "<bytes>").ndim
        3
    """
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if image is None:
        raise ImageLoadError(f"Archivo vacío o no es una imagen válida: {origin}")
    return image
