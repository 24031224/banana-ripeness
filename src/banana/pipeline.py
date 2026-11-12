"""Orquestador: encadena todas las etapas y es el único punto de entrada para app y scripts.

Flujo::

    BGRImage ─preprocess→ PreprocessedImage ─segment→ Segmentation
             ─extract_features→ FeatureSet ─classify→ Verdict

La app de Streamlit y los scripts SOLO deben importar desde aquí (y desde
``banana.common``), nunca llamar a las etapas por separado.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final, Self

import cv2
import numpy as np

from banana.acquisition import load_image
from banana.classifier import classify_rules
from banana.common import (
    BGRImage,
    FeatureSet,
    ImageLoadError,
    Mask,
    PipelineResult,
    Verdict,
)
from banana.config import DEFAULT_THRESHOLDS_PATH, Thresholds, load_thresholds
from banana.features import extract_features
from banana.preprocess import preprocess
from banana.segmentation import segment

OVERLAY_COLOR_BGR: Final[tuple[int, int, int]] = (0, 200, 0)
"""Color (BGR) con que se tiñe la máscara en la imagen de salida de :func:`analyze`."""

OVERLAY_ALPHA: Final[float] = 0.4
"""Opacidad del teñido de la máscara, en [0, 1]."""


class BananaPipeline:
    """Pipeline completo de clasificación configurado con un ``Thresholds``.

    Attributes:
        thresholds: Configuración usada por todas las etapas.

    Examples:
        >>> pipe = BananaPipeline.from_yaml()
        >>> pipe.classify_path("data/samples/comestible_01.jpg").label
        <Ripeness.COMESTIBLE: 'Comestible'>
    """

    def __init__(self, thresholds: Thresholds) -> None:
        """Crea el pipeline con una configuración ya cargada.

        Args:
            thresholds: Umbrales de todas las etapas.
        """
        self.thresholds = thresholds

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_THRESHOLDS_PATH) -> Self:
        """Construye el pipeline leyendo los umbrales desde un YAML.

        Args:
            path: Ruta al archivo de umbrales.

        Returns:
            Instancia lista para usarse.

        Raises:
            ConfigError: Si el YAML es inválido.

        Examples:
            >>> pipe = BananaPipeline.from_yaml("config/thresholds.yaml")
        """
        return cls(load_thresholds(path))

    def run(self, image: BGRImage) -> PipelineResult:
        """Ejecuta todas las etapas sobre una imagen y devuelve los resultados intermedios.

        Args:
            image: Imagen BGR cruda.

        Returns:
            ``PipelineResult`` con imagen original, preprocesada, segmentación y veredicto.

        Raises:
            SegmentationError: Si no se detecta un plátano.

        Examples:
            >>> result = pipe.run(load_image("data/samples/inmaduro_01.jpg"))
            >>> result.verdict.label
            <Ripeness.INMADURO: 'Inmaduro'>
        """
        pre = preprocess(image, self.thresholds.preprocess)
        seg = segment(pre.hsv, self.thresholds.segmentation)
        features = extract_features(pre.hsv, seg, self.thresholds.features)
        verdict = classify_rules(features, self.thresholds.classifier)
        return PipelineResult(original=image, preprocessed=pre, segmentation=seg, verdict=verdict)

    def classify_path(self, path: Path | str) -> Verdict:
        """Atajo: carga una imagen desde disco y devuelve solo el veredicto.

        Args:
            path: Ruta a la imagen.

        Returns:
            Veredicto de la imagen.

        Raises:
            ImageLoadError: Si la imagen no se puede leer.
            SegmentationError: Si no se detecta un plátano.

        Examples:
            >>> pipe.classify_path("data/samples/comestible_01.jpg").label
            <Ripeness.COMESTIBLE: 'Comestible'>
        """
        return self.run(load_image(path)).verdict


@lru_cache(maxsize=1)
def _default_pipeline() -> BananaPipeline:
    """Pipeline con el YAML por defecto, creado una sola vez por proceso."""
    return BananaPipeline.from_yaml()


def overlay_mask(image: BGRImage, mask: Mask) -> BGRImage:
    """Superpone la máscara sobre la imagen: teñido semitransparente y contorno.

    Args:
        image: Imagen BGR ``uint8``.
        mask: Máscara 0/255 con el mismo alto y ancho que ``image``.

    Returns:
        Copia de ``image`` con el plátano resaltado; ``image`` no se modifica.

    Raises:
        ValueError: Si la máscara y la imagen no tienen el mismo alto y ancho.

    Examples:
        >>> out = overlay_mask(img, mask)
        >>> out.shape == img.shape
        True
    """
    if mask.shape != image.shape[:2]:
        raise ValueError(f"Máscara {mask.shape} y imagen {image.shape[:2]} difieren de tamaño")
    color = np.empty_like(image)
    color[:] = OVERLAY_COLOR_BGR
    blended = cv2.addWeighted(image, 1.0 - OVERLAY_ALPHA, color, OVERLAY_ALPHA, 0.0)
    out = image.copy()
    inside = mask > 0
    out[inside] = blended[inside]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, OVERLAY_COLOR_BGR, 2)
    return out


def analyze(
    path_or_array: Path | str | BGRImage,
    pipeline: BananaPipeline | None = None,
) -> tuple[Verdict, FeatureSet, BGRImage]:
    """Clasifica una imagen y devuelve veredicto, características y vista con la máscara.

    Args:
        path_or_array: Ruta a una imagen o imagen BGR ``uint8`` HxWx3 ya cargada.
        pipeline: Pipeline a usar; por defecto se carga ``config/thresholds.yaml``
            (una sola vez por proceso).

    Returns:
        Tupla ``(verdict, features, overlay)``: el veredicto explicable, el
        ``FeatureSet`` usado y la imagen preprocesada (BGR) con la máscara superpuesta.

    Raises:
        ImageLoadError: Si la ruta no se puede leer o el arreglo no es una imagen
            BGR ``uint8`` de HxWx3.
        SegmentationError: Si no se detecta un plátano.

    Examples:
        >>> verdict, features, overlay = analyze("data/samples/comestible_01.jpg")
        >>> verdict.label
        <Ripeness.COMESTIBLE: 'Comestible'>
    """
    if isinstance(path_or_array, np.ndarray):
        image = path_or_array
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ImageLoadError("Se esperaba una imagen BGR uint8 de forma HxWx3")
    else:
        image = load_image(path_or_array)
    result = (pipeline or _default_pipeline()).run(image)
    overlay = overlay_mask(result.preprocessed.bgr, result.segmentation.mask)
    return result.verdict, result.verdict.features, overlay
