"""Carga y validación de umbrales desde ``config/thresholds.yaml``.

Regla del proyecto: **ningún umbral se escribe en el código**. Cada etapa recibe
su sección de configuración como dataclass tipada, lo que permite probarla de
forma aislada construyendo la config a mano en los tests.

Recordatorio: los valores H del YAML están en escala OpenCV (0-179).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from banana.common import HSVRange

DEFAULT_THRESHOLDS_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "config" / "thresholds.yaml"
)
"""Ruta por defecto al YAML (válida con la instalación editable ``pip install -e .``)."""

WhiteBalanceMethod = Literal["none", "gray_world"]


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Parámetros de :mod:`banana.preprocess`.

    Attributes:
        target_width: Ancho en píxeles al que se redimensiona (se conserva la proporción).
        blur_kernel: Tamaño del kernel gaussiano; impar y >= 1 (1 = sin suavizado).
        white_balance: Método de balance de blancos: ``"none"`` o ``"gray_world"``.
    """

    target_width: int
    blur_kernel: int
    white_balance: WhiteBalanceMethod


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    """Parámetros de :mod:`banana.segmentation`.

    Attributes:
        banana_hsv: Rango HSV que cubre la cáscara en TODAS sus madureces (verde a café).
        morph_kernel: Tamaño del elemento estructurante (impar).
        morph_iterations: Iteraciones de apertura/cierre morfológico.
        min_area_ratio: Área mínima del plátano respecto a la imagen, en [0, 1].
    """

    banana_hsv: HSVRange
    morph_kernel: int
    morph_iterations: int
    min_area_ratio: float


@dataclass(frozen=True, slots=True)
class FeaturesConfig:
    """Parámetros de :mod:`banana.features`.

    Attributes:
        hue_bins: Número de cubetas del histograma de H (cubren 0-179).
        green: Rango HSV considerado "verde".
        yellow: Rango HSV considerado "amarillo".
        brown: Rango HSV considerado "café".
        dark_spot_value_max: Píxeles con V por debajo de este valor cuentan como mancha.
    """

    hue_bins: int
    green: HSVRange
    yellow: HSVRange
    brown: HSVRange
    dark_spot_value_max: int


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    """Umbrales de las reglas de :mod:`banana.classifier`.

    Attributes:
        no_comestible_brown_ratio_min: ``brown_ratio`` mínimo para "No comestible".
        no_comestible_dark_ratio_min: ``dark_spot_ratio`` mínimo para "No comestible".
        inmaduro_green_ratio_min: ``green_ratio`` mínimo para "Inmaduro".
    """

    no_comestible_brown_ratio_min: float
    no_comestible_dark_ratio_min: float
    inmaduro_green_ratio_min: float


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Configuración completa del pipeline (raíz del YAML).

    Attributes:
        version: Versión del esquema del YAML.
        preprocess: Sección ``preprocess``.
        segmentation: Sección ``segmentation``.
        features: Sección ``features``.
        classifier: Sección ``classifier``.
    """

    version: int
    preprocess: PreprocessConfig
    segmentation: SegmentationConfig
    features: FeaturesConfig
    classifier: ClassifierConfig


def load_thresholds(path: Path | str = DEFAULT_THRESHOLDS_PATH) -> Thresholds:
    """Lee el YAML de umbrales y lo convierte en dataclasses validadas.

    Valida, como mínimo: que existan todas las secciones y claves, que H esté
    en 0-179 (escala OpenCV) y S/V en 0-255, que ``lower <= upper`` componente
    a componente, que los kernels sean impares y que las proporciones estén en [0, 1].

    Args:
        path: Ruta al archivo YAML. Por defecto ``config/thresholds.yaml``.

    Returns:
        Objeto ``Thresholds`` inmutable con todas las secciones.

    Raises:
        ConfigError: Si el archivo no existe, no es YAML válido o algún valor es inválido.

    Examples:
        >>> cfg = load_thresholds()
        >>> cfg.preprocess.target_width
        640
    """
    raise NotImplementedError
