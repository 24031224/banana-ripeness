"""Entrena un árbol de decisión, lo evalúa y lo compara con las reglas actuales.

Lee ``data/features_train.csv`` y ``data/features_test.csv`` (generados con
``scripts/extract_dataset.py``), entrena un ``DecisionTreeClassifier`` y guarda en
``docs/results/``:

* ``matriz_confusion_arbol.png``: matriz de confusión del árbol sobre test.
* ``reporte_arbol.md``: precision, recall y F1 por clase.
* ``importancia_caracteristicas.md``: importancia de cada característica.
* ``arbol.txt``: el árbol como texto (``export_text``).
* ``arbol.png``: el árbol dibujado (``plot_tree``).
* ``comparacion.md``: tabla reglas vs árbol sobre el mismo test, con el recall de
  ``no_comestible`` destacado.

Uso::

    python scripts/train_tree.py
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier, export_text

from banana.classifier import TREE_MAX_DEPTH, classify_rules
from banana.common import DATASET_FOLDERS, BananaError, FeatureSet
from banana.config import DEFAULT_THRESHOLDS_PATH, ClassifierConfig, load_thresholds
from extract_dataset import (
    DEFAULT_DATA_DIR,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    DatasetError,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

TREE_RANDOM_STATE: Final[int] = 42
"""Semilla del árbol, para que el entrenamiento sea reproducible."""

CLASS_ORDER: Final[tuple[str, ...]] = tuple(DATASET_FOLDERS)
"""Clases en orden fijo (nombres de carpeta): inmaduro, comestible, no_comestible."""

CRITICAL_CLASS: Final[str] = "no_comestible"
"""Clase cuyo recall se destaca: no detectarla es el error más costoso."""

DEFAULT_RESULTS_DIR: Final[Path] = Path("docs") / "results"
"""Carpeta de salida de los resultados."""

RESULT_FILES: Final[dict[str, str]] = {
    "confusion": "matriz_confusion_arbol.png",
    "report": "reporte_arbol.md",
    "importance": "importancia_caracteristicas.md",
    "text": "arbol.txt",
    "plot": "arbol.png",
    "comparison": "comparacion.md",
}
"""Nombre de cada archivo de resultados."""

METRIC_NAMES: Final[tuple[str, ...]] = ("precision", "recall", "f1")
"""Métricas por clase que se reportan."""


@dataclass(frozen=True, slots=True)
class Metrics:
    """Métricas de un clasificador sobre el conjunto de test.

    Attributes:
        accuracy: Exactitud global en [0, 1].
        per_class: Clase → ``{"precision", "recall", "f1", "support"}``.
        confusion: Matriz de confusión; filas = verdadero, columnas = predicho,
            en el orden de ``CLASS_ORDER``.
    """

    accuracy: float
    per_class: dict[str, dict[str, float]]
    confusion: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Dataset:
    """Características y etiquetas leídas de un CSV.

    Attributes:
        x: Matriz ``(n, len(FEATURE_COLUMNS))`` de características.
        y: Etiqueta de cada fila (nombre de carpeta de la clase).
    """

    x: npt.NDArray[np.float64]
    y: list[str]


def load_features_csv(path: Path) -> Dataset:
    """Lee un CSV generado por ``extract_dataset.py``.

    Args:
        path: Ruta al CSV.

    Returns:
        ``Dataset`` con las características y las etiquetas.

    Raises:
        DatasetError: Si el archivo no existe, faltan columnas, hay valores no
            numéricos, etiquetas desconocidas o no tiene filas.

    Examples:
        >>> ds = load_features_csv(Path("data/features_train.csv"))
        >>> ds.x.shape[1] == len(FEATURE_COLUMNS)
        True
    """
    if not path.is_file():
        raise DatasetError(
            f"No existe {path.as_posix()}. Genéralo primero con "
            "`python scripts/extract_dataset.py --split train` (y `--split test`).\n"
            "Estructura esperada del dataset:\n"
            + "\n".join(f"  data/<train|test>/{folder}/*.jpg|png" for folder in CLASS_ORDER)
        )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [
            c for c in (LABEL_COLUMN, *FEATURE_COLUMNS) if c not in (reader.fieldnames or [])
        ]
        if missing:
            raise DatasetError(f"Faltan columnas en {path.as_posix()}: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise DatasetError(f"{path.as_posix()} no tiene filas")
    unknown = sorted({r[LABEL_COLUMN] for r in rows} - set(CLASS_ORDER))
    if unknown:
        raise DatasetError(f"Etiquetas desconocidas en {path.as_posix()}: {', '.join(unknown)}")
    try:
        x = np.array([[float(r[c]) for c in FEATURE_COLUMNS] for r in rows], dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise DatasetError(f"Valor no numérico en {path.as_posix()}: {exc}") from exc
    return Dataset(x=x, y=[r[LABEL_COLUMN] for r in rows])


def fit_tree(train: Dataset) -> DecisionTreeClassifier:
    """Entrena el árbol de decisión sobre las características del CSV.

    Args:
        train: Datos de entrenamiento.

    Returns:
        ``DecisionTreeClassifier`` con profundidad máxima ``TREE_MAX_DEPTH``.

    Raises:
        DatasetError: Si hay menos de dos clases distintas.

    Examples:
        >>> tree = fit_tree(train)
        >>> tree.get_depth() <= 3
        True
    """
    if len(set(train.y)) < 2:
        raise DatasetError("Se necesitan al menos dos clases distintas para entrenar el árbol")
    tree = DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=TREE_RANDOM_STATE)
    return tree.fit(train.x, train.y)


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Metrics:
    """Calcula exactitud, precision/recall/F1 por clase y matriz de confusión.

    Args:
        y_true: Clases verdaderas.
        y_pred: Clases predichas.

    Returns:
        ``Metrics`` con las clases en el orden de ``CLASS_ORDER``.

    Examples:
        >>> compute_metrics(["inmaduro"], ["inmaduro"]).accuracy
        1.0
    """
    labels = list(CLASS_ORDER)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": float(support[i]),
        }
        for i, name in enumerate(labels)
    }
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        per_class=per_class,
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
    )


def row_to_features(row: npt.NDArray[np.float64]) -> FeatureSet:
    """Reconstruye un ``FeatureSet`` mínimo desde una fila de características del CSV.

    Solo se rellenan las proporciones que usan las reglas (verde, amarillo, marrón,
    oscuro y otros); los estadísticos HSV no se guardan en el CSV y quedan en 0.0.

    Args:
        row: Fila ordenada como ``FEATURE_COLUMNS``.

    Returns:
        ``FeatureSet`` apto para :func:`banana.classifier.classify_rules`.

    Examples:
        >>> row_to_features(np.array([0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1])).green_ratio
        0.8
    """
    values = dict(zip(FEATURE_COLUMNS, (float(v) for v in row), strict=True))
    return FeatureSet(
        hue_mean=0.0,
        hue_std=0.0,
        saturation_mean=0.0,
        value_mean=0.0,
        green_ratio=values["pct_verde"],
        yellow_ratio=values["pct_amarillo"],
        brown_ratio=values["pct_marron"],
        dark_spot_ratio=values["pct_oscuro"],
        hue_histogram=(),
        other_ratio=values["other_ratio"],
    )


def predict_rules(x: npt.NDArray[np.float64], cfg: ClassifierConfig) -> list[str]:
    """Clasifica cada fila con las reglas actuales (``classify_rules``).

    Args:
        x: Matriz de características ordenada como ``FEATURE_COLUMNS``.
        cfg: Sección ``classifier`` de los umbrales.

    Returns:
        Nombre de carpeta de la clase predicha para cada fila.

    Examples:
        >>> predict_rules(test.x, thresholds.classifier)[:1]
        ['comestible']
    """
    folder_of = {ripeness: folder for folder, ripeness in DATASET_FOLDERS.items()}
    return [folder_of[classify_rules(row_to_features(row), cfg).label] for row in x]


def feature_importances(tree: DecisionTreeClassifier) -> list[tuple[str, float]]:
    """Importancia de cada característica, de mayor a menor.

    Args:
        tree: Árbol entrenado con :func:`fit_tree`.

    Returns:
        Lista de ``(nombre, importancia)``; las importancias suman 1.0.

    Examples:
        >>> feature_importances(tree)[0][0] in FEATURE_COLUMNS
        True
    """
    pairs = zip(FEATURE_COLUMNS, (float(v) for v in tree.feature_importances_), strict=True)
    return sorted(pairs, key=lambda p: p[1], reverse=True)


def _md_table(header: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    """Arma una tabla Markdown."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def report_markdown(metrics: Metrics, title: str) -> str:
    """Genera el reporte precision/recall/F1 por clase en Markdown.

    Args:
        metrics: Métricas del clasificador.
        title: Título del reporte.

    Returns:
        Texto Markdown con la tabla por clase y la exactitud global.

    Examples:
        >>> report_markdown(compute_metrics(["inmaduro"], ["inmaduro"]), "Árbol")[:7]
        '# Árbol'
    """
    body = [
        [
            f"`{name}`",
            *(f"{metrics.per_class[name][m]:.3f}" for m in METRIC_NAMES),
            f"{int(metrics.per_class[name]['support'])}",
        ]
        for name in CLASS_ORDER
    ]
    table = _md_table(["Clase", "Precision", "Recall", "F1", "Soporte"], body)
    return f"# {title}\n\n{table}\n\n**Exactitud global:** {metrics.accuracy:.3f}\n"


def importance_markdown(tree: DecisionTreeClassifier) -> str:
    """Genera la tabla de importancia de características en Markdown.

    Args:
        tree: Árbol entrenado.

    Returns:
        Texto Markdown con una fila por característica.

    Examples:
        >>> importance_markdown(tree).startswith("# Importancia")
        True
    """
    body = [[f"`{n}`", f"{v:.3f}"] for n, v in feature_importances(tree)]
    return (
        "# Importancia de características\n\n"
        + _md_table(["Característica", "Importancia"], body)
        + "\n"
    )


def comparison_markdown(rules: Metrics, tree: Metrics) -> str:
    """Genera la tabla comparativa reglas vs árbol, destacando el recall de ``no_comestible``.

    Args:
        rules: Métricas de ``classify_rules`` sobre test.
        tree: Métricas del árbol sobre el mismo test.

    Returns:
        Texto Markdown con la conclusión sobre ``no_comestible`` y la tabla completa.

    Examples:
        >>> m = compute_metrics(["inmaduro"], ["inmaduro"])
        >>> "Reglas" in comparison_markdown(m, m)
        True
    """
    r_recall = rules.per_class[CRITICAL_CLASS]["recall"]
    t_recall = tree.per_class[CRITICAL_CLASS]["recall"]
    diff = t_recall - r_recall
    if diff == 0.0:
        verdict = "ambos detectan la misma proporción"
    elif diff > 0:
        verdict = f"el árbol detecta {diff:+.1%} más"
    else:
        verdict = f"las reglas detectan {-diff:.1%} más"
    body: list[list[str]] = []
    for name in CLASS_ORDER:
        for metric in METRIC_NAMES:
            r, t = rules.per_class[name][metric], tree.per_class[name][metric]
            cells = [f"`{name}`", metric, f"{r:.3f}", f"{t:.3f}", f"{t - r:+.3f}"]
            if name == CRITICAL_CLASS and metric == "recall":
                cells = [f"**{c}**" for c in cells]
            body.append(cells)
    body.append(
        [
            "Global",
            "exactitud",
            f"{rules.accuracy:.3f}",
            f"{tree.accuracy:.3f}",
            f"{tree.accuracy - rules.accuracy:+.3f}",
        ]
    )
    table = _md_table(["Clase", "Métrica", "Reglas", "Árbol", "Diferencia"], body)
    return (
        "# Comparación: reglas vs árbol de decisión\n\n"
        f"> **Recall de `{CRITICAL_CLASS}`:** reglas {r_recall:.3f} · árbol {t_recall:.3f} "
        f"({verdict}). Es la métrica crítica: un plátano no comestible clasificado como "
        "comestible es el error más costoso.\n\n"
        f"{table}\n"
    )


def _plot_confusion(metrics: Metrics, path: Path) -> None:
    """Guarda la matriz de confusión como PNG."""
    from matplotlib import pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(metrics.confusion, display_labels=list(CLASS_ORDER)).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Matriz de confusión - árbol (test)")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Verdadero")
    _save_figure(fig, path)


def _plot_tree(tree: DecisionTreeClassifier, path: Path) -> None:
    """Guarda el árbol dibujado con ``plot_tree`` como PNG."""
    from matplotlib import pyplot as plt
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        tree,
        feature_names=list(FEATURE_COLUMNS),
        class_names=[str(c) for c in tree.classes_],
        filled=True,
        rounded=True,
        ax=ax,
    )
    _save_figure(fig, path)


def _save_figure(fig: Figure, path: Path) -> None:
    """Guarda y cierra una figura de matplotlib."""
    from matplotlib import pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_results(
    tree: DecisionTreeClassifier, tree_metrics: Metrics, rules_metrics: Metrics, out_dir: Path
) -> list[Path]:
    """Escribe todos los archivos de resultados en ``out_dir``.

    Args:
        tree: Árbol entrenado.
        tree_metrics: Métricas del árbol sobre test.
        rules_metrics: Métricas de las reglas sobre test.
        out_dir: Carpeta de salida (se crea si no existe).

    Returns:
        Rutas de los archivos escritos.

    Examples:
        >>> paths = write_results(tree, tm, rm, Path("docs/results"))
    """
    import matplotlib

    matplotlib.use("Agg")  # sin ventana: solo se guardan archivos
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: out_dir / name for key, name in RESULT_FILES.items()}
    _plot_confusion(tree_metrics, paths["confusion"])
    _plot_tree(tree, paths["plot"])
    paths["report"].write_text(
        report_markdown(tree_metrics, "Reporte del árbol (test)"), encoding="utf-8"
    )
    paths["importance"].write_text(importance_markdown(tree), encoding="utf-8")
    paths["text"].write_text(
        export_text(tree, feature_names=list(FEATURE_COLUMNS)), encoding="utf-8"
    )
    paths["comparison"].write_text(
        comparison_markdown(rules_metrics, tree_metrics), encoding="utf-8"
    )
    return list(paths.values())


def run(
    train_csv: Path, test_csv: Path, out_dir: Path, cfg: ClassifierConfig
) -> tuple[Metrics, Metrics]:
    """Entrena, evalúa, compara y guarda los resultados.

    Args:
        train_csv: CSV de entrenamiento.
        test_csv: CSV de test.
        out_dir: Carpeta donde se guardan los resultados.
        cfg: Sección ``classifier`` de los umbrales (para las reglas).

    Returns:
        Tupla ``(métricas_del_árbol, métricas_de_las_reglas)`` sobre test.

    Raises:
        DatasetError: Si algún CSV falta o es inválido.

    Examples:
        >>> tree_m, rules_m = run(Path("a.csv"), Path("b.csv"), Path("out"), cfg)  # doctest: +SKIP
    """
    train = load_features_csv(train_csv)
    test = load_features_csv(test_csv)
    tree = fit_tree(train)
    tree_metrics = compute_metrics(test.y, [str(p) for p in tree.predict(test.x)])
    rules_metrics = compute_metrics(test.y, predict_rules(test.x, cfg))
    write_results(tree, tree_metrics, rules_metrics, out_dir)
    return tree_metrics, rules_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y lee los argumentos de línea de comandos.

    Args:
        argv: Lista de argumentos; ``None`` usa ``sys.argv``.

    Returns:
        Namespace con ``train_csv``, ``test_csv``, ``out_dir`` y ``config`` (todos Path).

    Examples:
        >>> parse_args([]).out_dir.name
        'results'
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_DATA_DIR / "features_train.csv")
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_DATA_DIR / "features_test.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del script.

    Args:
        argv: Argumentos de línea de comandos; ``None`` usa ``sys.argv``.

    Returns:
        Código de salida: 0 si todo salió bien, 1 si faltan datos o son inválidos.

    Examples:
        >>> main([])  # doctest: +SKIP
        0
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("train_tree")
    args = parse_args(argv)
    try:
        cfg = load_thresholds(args.config).classifier
        tree_metrics, rules_metrics = run(args.train_csv, args.test_csv, args.out_dir, cfg)
    except BananaError as exc:
        log.error("%s", exc)
        return 1
    log.info(
        "Recall %s - reglas: %.3f, árbol: %.3f. Resultados en %s",
        CRITICAL_CLASS,
        rules_metrics.per_class[CRITICAL_CLASS]["recall"],
        tree_metrics.per_class[CRITICAL_CLASS]["recall"],
        args.out_dir.as_posix(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
