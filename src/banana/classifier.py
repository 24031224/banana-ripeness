"""Clasificadores explicables sobre un :class:`banana.common.FeatureSet`.

Hay dos clasificadores con la misma salida (:class:`banana.common.Verdict`):

* **Reglas** (:func:`classify_rules`, método principal). Orden de evaluación, la
  primera regla que se cumple gana:

  1. **No comestible** si ``brown_ratio >= no_comestible_brown_ratio_min``
     o ``dark_spot_ratio >= no_comestible_dark_ratio_min``.
  2. **Inmaduro** si ``green_ratio >= inmaduro_green_ratio_min``.
  3. **Comestible** en cualquier otro caso.

  Todos los umbrales vienen de ``config/thresholds.yaml`` (sección ``classifier``).

* **Árbol de decisión** (:func:`train_tree`, :func:`classify_tree`,
  :func:`export_tree_rules`), de profundidad máxima 3, entrenado con un CSV de
  características etiquetadas. Sirve para comparar contra las reglas y para
  inspirar umbrales; sus reglas se pueden leer como texto.

Recordatorio: ``hue_mean`` y ``hue_std`` están en escala OpenCV (H en 0-179).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from banana.common import DATASET_FOLDERS, BananaError, FeatureSet, Ripeness, Verdict
from banana.config import ClassifierConfig

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeClassifier

TREE_FEATURES: Final[tuple[str, ...]] = (
    "hue_mean",  # H en escala OpenCV 0-179
    "hue_std",  # H en escala OpenCV 0-179
    "saturation_mean",
    "value_mean",
    "green_ratio",
    "yellow_ratio",
    "brown_ratio",
    "dark_spot_ratio",
)
"""Campos escalares de ``FeatureSet`` que usa el árbol, en orden fijo."""

LABEL_COLUMN: Final[str] = "label"
"""Nombre de la columna de etiqueta en el CSV de entrenamiento."""

TREE_MAX_DEPTH: Final[int] = 3
"""Profundidad máxima del árbol (pedida por el equipo para que sea legible)."""

TREE_RANDOM_STATE: Final[int] = 0
"""Semilla para que el entrenamiento sea reproducible."""


class ClassifierError(BananaError):
    """El CSV de entrenamiento es inválido o el modelo no se puede usar."""


# --------------------------------------------------------------------------- #
# Reglas
# --------------------------------------------------------------------------- #
def margin_confidence(value: float, threshold: float) -> float:
    """Convierte la distancia entre un valor y su umbral en una confianza en [0, 1].

    Cuanto más lejos está ``value`` del umbral que decidió la clase, más alta la
    confianza. Es simétrica respecto al umbral (da igual estar por encima o por
    debajo) y se normaliza con la mayor distancia posible para una proporción en
    [0, 1], de modo que satura en 1.0.

    Args:
        value: Valor de la característica (p. ej. ``brown_ratio``), en [0, 1].
        threshold: Umbral de la regla, en (0, 1].

    Returns:
        Confianza en [0, 1]; 0.0 exactamente en el umbral.

    Raises:
        ValueError: Si ``threshold`` no está en (0, 1].

    Examples:
        >>> margin_confidence(0.40, 0.40)
        0.0
        >>> margin_confidence(1.0, 0.5)
        1.0
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold debe estar en (0, 1], se recibió {threshold}")
    clipped = min(max(value, 0.0), 1.0)
    max_distance = max(threshold, 1.0 - threshold)
    return min(abs(clipped - threshold) / max_distance, 1.0)


def classify_rules(features: FeatureSet, cfg: ClassifierConfig) -> Verdict:
    """Aplica las reglas en orden y devuelve un veredicto explicable.

    Prioridad: No comestible > Inmaduro > Comestible. La confianza es el margen
    (:func:`margin_confidence`) de la regla decisiva; si dispararon varias reglas
    de "No comestible" se usa la de mayor margen. Para "Comestible", que se asigna
    cuando ninguna regla se cumple, es el menor margen de las tres reglas
    descartadas (qué tan cerca estuvo de cambiar de clase).

    Args:
        features: Características del plátano.
        cfg: Sección ``classifier`` de los umbrales.

    Returns:
        ``Verdict`` con la clase, la confianza, las razones (texto con valor y umbral
        de cada regla que decidió) y las características usadas.

    Examples:
        >>> v = classify_rules(fs, thresholds.classifier)
        >>> v.label in Ripeness
        True
    """
    brown, dark, green = features.brown_ratio, features.dark_spot_ratio, features.green_ratio
    brown_min = cfg.no_comestible_brown_ratio_min
    dark_min = cfg.no_comestible_dark_ratio_min
    green_min = cfg.inmaduro_green_ratio_min

    # Regla 1: No comestible (cualquiera de las dos condiciones).
    reasons: list[str] = []
    margins: list[float] = []
    if brown >= brown_min:
        reasons.append(f"brown_ratio={brown:.2f} >= {brown_min:.2f}")
        margins.append(margin_confidence(brown, brown_min))
    if dark >= dark_min:
        reasons.append(f"dark_spot_ratio={dark:.2f} >= {dark_min:.2f}")
        margins.append(margin_confidence(dark, dark_min))
    if reasons:
        return Verdict(Ripeness.NO_COMESTIBLE, max(margins), tuple(reasons), features)

    # Regla 2: Inmaduro.
    if green >= green_min:
        return Verdict(
            Ripeness.INMADURO,
            margin_confidence(green, green_min),
            (f"green_ratio={green:.2f} >= {green_min:.2f}",),
            features,
        )

    # Regla 3: Comestible por descarte; se explica con las reglas que NO se cumplieron.
    return Verdict(
        Ripeness.COMESTIBLE,
        min(
            margin_confidence(brown, brown_min),
            margin_confidence(dark, dark_min),
            margin_confidence(green, green_min),
        ),
        (
            f"brown_ratio={brown:.2f} < {brown_min:.2f}",
            f"dark_spot_ratio={dark:.2f} < {dark_min:.2f}",
            f"green_ratio={green:.2f} < {green_min:.2f}",
        ),
        features,
    )


# --------------------------------------------------------------------------- #
# Árbol de decisión
# --------------------------------------------------------------------------- #
def feature_vector(features: FeatureSet) -> list[float]:
    """Extrae los campos escalares de un ``FeatureSet`` en el orden de ``TREE_FEATURES``.

    Args:
        features: Características del plátano.

    Returns:
        Lista de floats, uno por cada nombre de ``TREE_FEATURES``.

    Examples:
        >>> len(feature_vector(fs)) == len(TREE_FEATURES)
        True
    """
    return [float(getattr(features, name)) for name in TREE_FEATURES]


def _parse_label(raw: str) -> Ripeness:
    """Interpreta una etiqueta del CSV (``"Inmaduro"`` o carpeta ``"no_comestible"``)."""
    text = raw.strip()
    if text in DATASET_FOLDERS:
        return DATASET_FOLDERS[text]
    try:
        return Ripeness(text)
    except ValueError:
        raise ClassifierError(f"Etiqueta desconocida en el CSV: {raw!r}") from None


def train_tree(csv_path: Path | str) -> DecisionTreeClassifier:
    """Entrena un ``DecisionTreeClassifier(max_depth=3)`` desde un CSV etiquetado.

    El CSV debe tener encabezado con una columna ``label`` (nombre de la clase,
    ``"Inmaduro"``/``"Comestible"``/``"No comestible"``, o el de la carpeta del
    dataset: ``inmaduro``/``comestible``/``no_comestible``) y una columna por cada
    nombre de ``TREE_FEATURES``. Columnas extra se ignoran.

    Args:
        csv_path: Ruta al CSV de entrenamiento.

    Returns:
        Árbol entrenado; ``classes_`` contiene los valores de ``Ripeness`` (texto).

    Raises:
        ClassifierError: Si el archivo no existe, faltan columnas, hay valores no
            numéricos, etiquetas desconocidas o menos de dos clases distintas.

    Examples:
        >>> tree = train_tree("data/features.csv")
        >>> tree.get_depth() <= 3
        True
    """
    from sklearn.tree import DecisionTreeClassifier

    path = Path(csv_path)
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [
                c for c in (*TREE_FEATURES, LABEL_COLUMN) if c not in (reader.fieldnames or [])
            ]
            if missing:
                raise ClassifierError(f"Faltan columnas en {path}: {', '.join(missing)}")
            rows = list(reader)
    except OSError as exc:
        raise ClassifierError(f"No se pudo leer {path}: {exc}") from exc

    try:
        x = np.array([[float(row[name]) for name in TREE_FEATURES] for row in rows], dtype=float)
    except (ValueError, TypeError) as exc:
        raise ClassifierError(f"Valor no numérico en {path}: {exc}") from exc
    y = [_parse_label(row[LABEL_COLUMN]).value for row in rows]

    if len(set(y)) < 2:
        raise ClassifierError("Se necesitan al menos dos clases distintas para entrenar el árbol")

    tree = DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=TREE_RANDOM_STATE)
    tree.fit(x, y)
    return tree


def classify_tree(tree: DecisionTreeClassifier, features: FeatureSet) -> Verdict:
    """Clasifica con un árbol entrenado y explica el camino de decisión.

    La confianza es la proporción de muestras de entrenamiento de la clase
    ganadora en la hoja alcanzada (``predict_proba``). Las razones son las
    comparaciones del camino raíz → hoja, p. ej. ``"brown_ratio=0.52 > 0.31"``.

    Args:
        tree: Árbol devuelto por :func:`train_tree`.
        features: Características del plátano.

    Returns:
        ``Verdict`` con clase, confianza y el camino recorrido como razones.

    Examples:
        >>> classify_tree(tree, fs).label in Ripeness
        True
    """
    sample = np.array([feature_vector(features)], dtype=float)
    proba = tree.predict_proba(sample)[0]
    best = int(np.argmax(proba))
    label = Ripeness(str(tree.classes_[best]))

    inner = tree.tree_
    reasons: list[str] = []
    for node in tree.decision_path(sample).indices:
        if inner.children_left[node] == inner.children_right[node]:  # hoja: sin condición
            continue
        name = TREE_FEATURES[inner.feature[node]]
        value = sample[0, inner.feature[node]]
        threshold = inner.threshold[node]
        op = "<=" if value <= threshold else ">"
        reasons.append(f"{name}={value:.2f} {op} {threshold:.2f}")
    if not reasons:  # árbol de un solo nodo
        reasons.append("árbol sin divisiones: clase mayoritaria del entrenamiento")

    return Verdict(label, float(proba[best]), tuple(reasons), features)


def export_tree_rules(tree: DecisionTreeClassifier) -> str:
    """Exporta las reglas del árbol como texto indentado legible.

    Args:
        tree: Árbol devuelto por :func:`train_tree`.

    Returns:
        Texto multilínea con las condiciones sobre ``TREE_FEATURES`` y la clase de
        cada hoja (formato de ``sklearn.tree.export_text``).

    Examples:
        >>> print(export_tree_rules(tree))  # doctest: +SKIP
        |--- brown_ratio <= 0.40
        |   |--- class: Comestible
    """
    from sklearn.tree import export_text

    return str(export_text(tree, feature_names=list(TREE_FEATURES)))
