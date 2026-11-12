"""Clasificador basado en reglas explicables sobre un :class:`banana.common.FeatureSet`.

Orden de evaluación (la primera regla que se cumple gana):

1. **No comestible** si ``brown_ratio >= no_comestible_brown_ratio_min``
   o ``dark_spot_ratio >= no_comestible_dark_ratio_min``.
2. **Inmaduro** si ``green_ratio >= inmaduro_green_ratio_min``.
3. **Comestible** en cualquier otro caso.

Todos los umbrales vienen de ``config/thresholds.yaml`` (sección ``classifier``).
"""

from __future__ import annotations

from banana.common import FeatureSet, Verdict
from banana.config import ClassifierConfig


def margin_confidence(value: float, threshold: float) -> float:
    """Convierte la distancia entre un valor y su umbral en una confianza en [0, 1].

    Cuanto más lejos está ``value`` del umbral que decidió la clase, más alta la
    confianza. Debe ser simétrica respecto al umbral y saturar en 1.0.

    Args:
        value: Valor de la característica (p. ej. ``brown_ratio``).
        threshold: Umbral de la regla, en (0, 1].

    Returns:
        Confianza en [0, 1]; 0.0 exactamente en el umbral.

    Raises:
        ValueError: Si ``threshold`` no está en (0, 1].

    Examples:
        >>> margin_confidence(0.40, 0.40)
        0.0
    """
    raise NotImplementedError


def classify(features: FeatureSet, cfg: ClassifierConfig) -> Verdict:
    """Aplica las reglas en orden y devuelve un veredicto explicable.

    Args:
        features: Características del plátano.
        cfg: Sección ``classifier`` de los umbrales.

    Returns:
        ``Verdict`` con la clase, la confianza, las razones (texto con valor y umbral
        de cada regla evaluada) y las características usadas.

    Examples:
        >>> v = classify(fs, thresholds.classifier)
        >>> v.label in Ripeness
        True
    """
    raise NotImplementedError
