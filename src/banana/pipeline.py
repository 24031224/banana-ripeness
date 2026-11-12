"""Orquestador: encadena todas las etapas y es el único punto de entrada para app y scripts.

Flujo::

    BGRImage ─preprocess→ PreprocessedImage ─segment→ Segmentation
             ─extract_features→ FeatureSet ─classify→ Verdict

La app de Streamlit y los scripts SOLO deben importar desde aquí (y desde
``banana.common``), nunca llamar a las etapas por separado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from banana.common import BGRImage, PipelineResult, Verdict
from banana.config import DEFAULT_THRESHOLDS_PATH, Thresholds


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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
