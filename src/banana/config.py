"""Carga y validación de umbrales desde ``config/thresholds.yaml``.

Regla del proyecto: **ningún umbral se escribe en el código**. Cada etapa recibe
su sección de configuración como dataclass tipada, lo que permite probarla de
forma aislada construyendo la config a mano en los tests.

Recordatorio: los valores H del YAML están en escala OpenCV (0-179).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, get_args

import yaml

from banana.common import ConfigError, HSVRange

DEFAULT_THRESHOLDS_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "config" / "thresholds.yaml"
)
"""Ruta por defecto al YAML (válida con la instalación editable ``pip install -e .``)."""

WhiteBalanceMethod = Literal["none", "gray_world"]

H_MAX: Final[int] = 179
"""Máximo de H en OpenCV (grados / 2; ¡no 360!)."""

SV_MAX: Final[int] = 255
"""Máximo de S y V en OpenCV."""


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
    yaml_path = Path(path)
    try:
        text = yaml_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"No existe el archivo de umbrales: {yaml_path}") from exc
    except OSError as exc:
        raise ConfigError(f"No se pudo leer {yaml_path}: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML inválido en {yaml_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{yaml_path} debe contener un mapa en la raíz")

    return Thresholds(
        version=_get_int(raw, "version", "<raíz>", minimum=1),
        preprocess=_parse_preprocess(_get_section(raw, "preprocess")),
        segmentation=_parse_segmentation(_get_section(raw, "segmentation")),
        features=_parse_features(_get_section(raw, "features")),
        classifier=_parse_classifier(_get_section(raw, "classifier")),
    )


# --------------------------------------------------------------------------- #
# Parseo de secciones
# --------------------------------------------------------------------------- #
def _parse_preprocess(sec: Mapping[str, Any]) -> PreprocessConfig:
    """Construye ``PreprocessConfig`` validando la sección ``preprocess``."""
    where = "preprocess"
    method = _get_value(sec, "white_balance", where)
    allowed = get_args(WhiteBalanceMethod)
    if method not in allowed:
        raise ConfigError(f"{where}.white_balance={method!r} no es válido; use uno de {allowed}")
    return PreprocessConfig(
        target_width=_get_int(sec, "target_width", where, minimum=1),
        blur_kernel=_get_odd(sec, "blur_kernel", where),
        white_balance=method,
    )


def _parse_segmentation(sec: Mapping[str, Any]) -> SegmentationConfig:
    """Construye ``SegmentationConfig`` validando la sección ``segmentation``."""
    where = "segmentation"
    return SegmentationConfig(
        banana_hsv=_get_hsv_range(sec, "banana_hsv", where),
        morph_kernel=_get_odd(sec, "morph_kernel", where),
        morph_iterations=_get_int(sec, "morph_iterations", where, minimum=1),
        min_area_ratio=_get_ratio(sec, "min_area_ratio", where),
    )


def _parse_features(sec: Mapping[str, Any]) -> FeaturesConfig:
    """Construye ``FeaturesConfig`` validando la sección ``features``."""
    where = "features"
    return FeaturesConfig(
        hue_bins=_get_int(sec, "hue_bins", where, minimum=1, maximum=H_MAX + 1),
        green=_get_hsv_range(sec, "green", where),
        yellow=_get_hsv_range(sec, "yellow", where),
        brown=_get_hsv_range(sec, "brown", where),
        dark_spot_value_max=_get_int(sec, "dark_spot_value_max", where, minimum=0, maximum=SV_MAX),
    )


def _parse_classifier(sec: Mapping[str, Any]) -> ClassifierConfig:
    """Construye ``ClassifierConfig`` validando la sección ``classifier``."""
    where = "classifier"
    return ClassifierConfig(
        no_comestible_brown_ratio_min=_get_ratio(sec, "no_comestible_brown_ratio_min", where),
        no_comestible_dark_ratio_min=_get_ratio(sec, "no_comestible_dark_ratio_min", where),
        inmaduro_green_ratio_min=_get_ratio(sec, "inmaduro_green_ratio_min", where),
    )


# --------------------------------------------------------------------------- #
# Helpers de validación (todos lanzan ConfigError con la ruta de la clave)
# --------------------------------------------------------------------------- #
def _get_value(sec: Mapping[str, Any], key: str, where: str) -> Any:  # noqa: ANN401
    """Devuelve ``sec[key]`` o lanza ``ConfigError`` si falta."""
    if key not in sec:
        raise ConfigError(f"Falta la clave '{where}.{key}' en el YAML de umbrales")
    return sec[key]


def _get_section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Devuelve la sección ``name`` y exige que sea un mapa."""
    sec = _get_value(raw, name, "<raíz>")
    if not isinstance(sec, Mapping):
        raise ConfigError(f"La sección '{name}' debe ser un mapa clave: valor")
    return sec


def _get_int(
    sec: Mapping[str, Any],
    key: str,
    where: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Lee un entero (rechaza bool y floats) dentro de ``[minimum, maximum]``."""
    value = _get_value(sec, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}.{key} debe ser un entero, no {value!r}")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ConfigError(f"{where}.{key}={value} fuera del rango [{minimum}, {maximum}]")
    return value


def _get_odd(sec: Mapping[str, Any], key: str, where: str) -> int:
    """Lee un entero impar >= 1 (tamaños de kernel)."""
    value = _get_int(sec, key, where, minimum=1)
    if value % 2 == 0:
        raise ConfigError(f"{where}.{key}={value} debe ser impar")
    return value


def _get_ratio(sec: Mapping[str, Any], key: str, where: str) -> float:
    """Lee una proporción numérica en [0, 1]."""
    value = _get_value(sec, key, where)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{where}.{key} debe ser un número, no {value!r}")
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{where}.{key}={value} debe estar en [0, 1]")
    return float(value)


def _get_hsv_triplet(value: Any, label: str) -> tuple[int, int, int]:  # noqa: ANN401
    """Valida una terna ``[H, S, V]``: H en 0-179 (OpenCV), S y V en 0-255."""
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ConfigError(f"{label} debe ser una lista [H, S, V], no {value!r}")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        raise ConfigError(f"{label} debe contener enteros, no {value!r}")
    h, s, v = value
    # H de OpenCV: 0-179 (grados / 2), nunca 0-360.
    if not 0 <= h <= H_MAX:
        raise ConfigError(f"{label}: H={h} fuera de 0-{H_MAX} (OpenCV usa grados/2, no 0-360)")
    if not 0 <= s <= SV_MAX or not 0 <= v <= SV_MAX:
        raise ConfigError(f"{label}: S={s}, V={v} deben estar en 0-{SV_MAX}")
    return (h, s, v)


def _get_hsv_range(sec: Mapping[str, Any], key: str, where: str) -> HSVRange:
    """Lee un rango ``{lower, upper}`` de ternas HSV y valida ``lower <= upper``."""
    label = f"{where}.{key}"
    rng = _get_value(sec, key, where)
    if not isinstance(rng, Mapping):
        raise ConfigError(f"{label} debe ser un mapa con 'lower' y 'upper'")
    lower = _get_hsv_triplet(_get_value(rng, "lower", label), f"{label}.lower")
    upper = _get_hsv_triplet(_get_value(rng, "upper", label), f"{label}.upper")
    if any(lo > up for lo, up in zip(lower, upper, strict=True)):
        raise ConfigError(f"{label}: lower {lower} debe ser <= upper {upper} en cada componente")
    return HSVRange(lower=lower, upper=upper)
