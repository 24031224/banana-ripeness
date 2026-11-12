"""Calibración de umbrales a partir de un dataset etiquetado.

Calcula el ``FeatureSet`` de cada imagen, resume la distribución de cada
característica por clase y propone umbrales para ``config/thresholds.yaml``.
El script NO sobrescribe el YAML: imprime la propuesta (o la guarda en otro
archivo) para que el equipo la revise y la suba en un commit propio.

Uso::

    python scripts/calibrate.py --dataset data/raw --output data/calibracion.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from banana.common import FeatureSet, Ripeness
from banana.config import Thresholds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y lee los argumentos de línea de comandos.

    Args:
        argv: Lista de argumentos; ``None`` usa ``sys.argv``.

    Returns:
        Namespace con ``dataset`` (Path), ``config`` (Path) y ``output`` (Path | None).

    Examples:
        >>> args = parse_args(["--dataset", "data/raw"])
    """
    raise NotImplementedError


def collect_features(
    dataset_root: Path, thresholds: Thresholds
) -> dict[Ripeness, list[FeatureSet]]:
    """Ejecuta preprocesamiento, segmentación y extracción sobre todo el dataset.

    Las imágenes donde falla la segmentación se reportan y se omiten.

    Args:
        dataset_root: Carpeta con subcarpetas por clase.
        thresholds: Umbrales actuales (para preprocesar y segmentar).

    Returns:
        Diccionario clase → lista de características.

    Examples:
        >>> por_clase = collect_features(Path("data/raw"), thresholds)
    """
    raise NotImplementedError


def suggest_thresholds(
    features_by_class: dict[Ripeness, list[FeatureSet]],
) -> dict[str, float]:
    """Propone umbrales de la sección ``classifier`` a partir de las distribuciones.

    Por ejemplo, ``inmaduro_green_ratio_min`` como el punto que mejor separa
    ``green_ratio`` de "Inmaduro" frente a "Comestible".

    Args:
        features_by_class: Salida de :func:`collect_features`.

    Returns:
        Diccionario con las mismas claves que la sección ``classifier`` del YAML.

    Examples:
        >>> propuesta = suggest_thresholds(por_clase)
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del script.

    Args:
        argv: Argumentos de línea de comandos; ``None`` usa ``sys.argv``.

    Returns:
        Código de salida (0 = éxito).

    Examples:
        >>> main(["--dataset", "data/raw"])
        0
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
