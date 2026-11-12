"""Tipos compartidos entre todos los módulos del clasificador.

Este módulo es el **contrato** del proyecto: define las estructuras de datos que
viajan entre etapas del pipeline. Cualquier cambio aquí afecta a todo el equipo,
así que se acuerda en grupo antes de hacerlo (ver ``docs/arquitectura.md``).

Convenciones de color:
    - Toda imagen color en memoria está en **BGR** (convención de OpenCV), nunca RGB.
    - En HSV, OpenCV usa **H en 0-179** (grados / 2, NO 0-360), S en 0-255 y V en 0-255.
    - Las máscaras son ``uint8`` con valores 0 (fondo) o 255 (plátano).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

BGRImage: TypeAlias = npt.NDArray[np.uint8]
"""Imagen color HxWx3, ``uint8``, canales en orden BGR."""

HSVImage: TypeAlias = npt.NDArray[np.uint8]
"""Imagen HxWx3 ``uint8`` en HSV de OpenCV: H en 0-179 (¡no 0-360!), S y V en 0-255."""

Mask: TypeAlias = npt.NDArray[np.uint8]
"""Máscara binaria HxW ``uint8`` con valores 0 (fondo) o 255 (plátano)."""

BoundingBox: TypeAlias = tuple[int, int, int, int]
"""Caja envolvente ``(x, y, ancho, alto)`` en píxeles, como ``cv2.boundingRect``."""


class Ripeness(StrEnum):
    """Clases de madurez que puede emitir el clasificador.

    El valor de cada miembro es la etiqueta exacta que se muestra al usuario.

    Examples:
        >>> Ripeness.NO_COMESTIBLE.value
        'No comestible'
    """

    INMADURO = "Inmaduro"
    COMESTIBLE = "Comestible"
    NO_COMESTIBLE = "No comestible"


DATASET_FOLDERS: dict[str, Ripeness] = {
    "inmaduro": Ripeness.INMADURO,
    "comestible": Ripeness.COMESTIBLE,
    "no_comestible": Ripeness.NO_COMESTIBLE,
}
"""Subcarpeta del dataset → clase. Estructura esperada: ``<raíz>/<carpeta>/*.jpg``."""


# --------------------------------------------------------------------------- #
# Excepciones del dominio
# --------------------------------------------------------------------------- #
class BananaError(Exception):
    """Excepción base de todos los errores del paquete ``banana``."""


class ImageLoadError(BananaError):
    """No se pudo leer o decodificar una imagen (archivo, bytes o cámara)."""


class SegmentationError(BananaError):
    """No se encontró un plátano válido en la imagen (máscara vacía o muy pequeña)."""


class ConfigError(BananaError):
    """El archivo de umbrales falta, está mal formado o tiene valores inválidos."""


# --------------------------------------------------------------------------- #
# Estructuras de datos que fluyen por el pipeline
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class HSVRange:
    """Rango HSV inclusivo, listo para usarse con ``cv2.inRange``.

    Recordatorio: en OpenCV H va de 0 a 179. Se exige ``lower[0] <= upper[0]``;
    los tonos que cruzan el 0 (rojos) requieren dos rangos separados.

    Attributes:
        lower: Límite inferior ``(H, S, V)``.
        upper: Límite superior ``(H, S, V)``.

    Examples:
        >>> amarillo = HSVRange(lower=(20, 60, 100), upper=(35, 255, 255))
    """

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class LabeledSample:
    """Imagen del dataset con su etiqueta verdadera (para calibrar y evaluar).

    Attributes:
        path: Ruta al archivo de imagen.
        label: Clase verdadera, deducida de la carpeta (ver ``DATASET_FOLDERS``).
    """

    path: Path
    label: Ripeness


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    """Salida de :func:`banana.preprocess.preprocess`.

    Attributes:
        bgr: Imagen BGR redimensionada, suavizada y con balance de blancos.
        hsv: La misma imagen en HSV de OpenCV (H en 0-179).
        scale: Factor aplicado al redimensionar (``nuevo_ancho / ancho_original``).
    """

    bgr: BGRImage
    hsv: HSVImage
    scale: float


@dataclass(frozen=True, slots=True)
class Segmentation:
    """Salida de :func:`banana.segmentation.segment`.

    Attributes:
        mask: Máscara binaria (0/255) del plátano, mismo alto y ancho que la imagen.
        bbox: Caja envolvente del plátano ``(x, y, ancho, alto)``.
        area_px: Número de píxeles con valor 255 en ``mask``.
        area_ratio: ``area_px`` dividido entre el total de píxeles, en [0, 1].
    """

    mask: Mask
    bbox: BoundingBox
    area_px: int
    area_ratio: float


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Vector de características de color calculado SOLO sobre la máscara.

    Todas las proporciones son fracciones en [0, 1] respecto a los píxeles
    del plátano (no de la imagen completa).

    Attributes:
        hue_mean: Media de H en escala OpenCV (0-179).
        hue_std: Desviación estándar de H (escala 0-179).
        saturation_mean: Media de S (0-255).
        value_mean: Media de V (0-255).
        green_ratio: Fracción de píxeles dentro del rango "verde".
        yellow_ratio: Fracción de píxeles dentro del rango "amarillo".
        brown_ratio: Fracción de píxeles dentro del rango "café".
        dark_spot_ratio: Fracción de píxeles con V menor al umbral de manchas.
        hue_histogram: Histograma de H normalizado (suma 1) con ``hue_bins``
            cubetas sobre 0-179.
        other_ratio: Fracción de píxeles de la máscara que no caen en verde,
            amarillo ni marrón (unión de los tres rangos, sin doble conteo).
            Vale 0.0 por defecto.
    """

    hue_mean: float
    hue_std: float
    saturation_mean: float
    value_mean: float
    green_ratio: float
    yellow_ratio: float
    brown_ratio: float
    dark_spot_ratio: float
    hue_histogram: tuple[float, ...]
    other_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class Verdict:
    """Decisión final del clasificador, explicable.

    Attributes:
        label: Clase asignada.
        confidence: Confianza heurística en [0, 1] (margen respecto al umbral decisivo).
        reasons: Frases legibles con las reglas que dispararon la decisión,
            p. ej. ``("brown_ratio=0.52 >= 0.40",)``.
        features: Características que se usaron para decidir.
    """

    label: Ripeness
    confidence: float
    reasons: tuple[str, ...]
    features: FeatureSet


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Resultado completo de una ejecución del pipeline (para depurar y para la app).

    Attributes:
        original: Imagen BGR tal como se recibió.
        preprocessed: Imagen tras el preprocesamiento.
        segmentation: Máscara y geometría del plátano.
        verdict: Clasificación final (incluye el ``FeatureSet``).
    """

    original: BGRImage
    preprocessed: PreprocessedImage
    segmentation: Segmentation
    verdict: Verdict
