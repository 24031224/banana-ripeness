"""Extrae las características de un dataset etiquetado y las guarda en un CSV.

Recorre ``data/<split>/<clase>/*.jpg|png``, ejecuta el pipeline hasta las
características y escribe ``data/features_<split>.csv``. Las imágenes cuya
segmentación (o lectura) falla se registran en ``data/extract_<split>.log`` y el
proceso continúa con las demás.

Uso::

    python scripts/extract_dataset.py --split train
    python scripts/extract_dataset.py --split test

Todas las proporciones (``pct_*``, ``area_manchas_pct`` y ``other_ratio``) son
fracciones en [0, 1] respecto a los píxeles del plátano, como en ``FeatureSet``.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from banana.acquisition import iter_dataset, load_image
from banana.common import DATASET_FOLDERS, BananaError, HSVImage, Mask
from banana.config import DEFAULT_THRESHOLDS_PATH
from banana.pipeline import BananaPipeline

SPLITS: Final[tuple[str, ...]] = ("train", "test")
"""Particiones válidas del dataset."""

DEFAULT_DATA_DIR: Final[Path] = Path("data")
"""Carpeta raíz de los datos (ignorada por git salvo ``data/samples``)."""

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "pct_verde",
    "pct_amarillo",
    "pct_marron",
    "pct_oscuro",
    "n_manchas",
    "area_manchas_pct",
    "other_ratio",
)
"""Columnas numéricas del CSV, en orden fijo (las usa también ``train_tree.py``)."""

PATH_COLUMN: Final[str] = "ruta"
"""Columna con la ruta de la imagen."""

LABEL_COLUMN: Final[str] = "etiqueta"
"""Columna con la etiqueta (nombre de la carpeta: ``inmaduro``, ``comestible``, ...)."""

CSV_COLUMNS: Final[tuple[str, ...]] = (PATH_COLUMN, LABEL_COLUMN, *FEATURE_COLUMNS)
"""Encabezado completo del CSV."""


class DatasetError(BananaError):
    """El dataset no existe, está vacío o no se pudo extraer ninguna característica."""


@dataclass(frozen=True, slots=True)
class SpotStats:
    """Manchas oscuras detectadas dentro del plátano.

    Attributes:
        count: Número de manchas (componentes conexas).
        area_ratio: Área total de las manchas respecto al área del plátano, en [0, 1].
    """

    count: int
    area_ratio: float


def expected_structure(split: str, data_dir: Path = DEFAULT_DATA_DIR) -> str:
    """Describe la estructura de carpetas que espera el script.

    Args:
        split: Partición (``train`` o ``test``).
        data_dir: Carpeta raíz de los datos.

    Returns:
        Texto multilínea con el árbol de carpetas esperado.

    Examples:
        >>> "no_comestible" in expected_structure("train")
        True
    """
    root = (data_dir / split).as_posix()
    lines = [f"  {root}/"]
    lines += [f"    {folder}/*.jpg|png" for folder in DATASET_FOLDERS]
    return "\n".join(lines)


def count_dark_spots(hsv: HSVImage, mask: Mask, value_max: int, morph_kernel: int) -> SpotStats:
    """Cuenta las manchas oscuras (V < ``value_max``) dentro de la máscara.

    Se aplica una apertura morfológica con el mismo kernel que usa la segmentación
    para descartar píxeles sueltos (ruido) que no son manchas reales.

    Args:
        hsv: Imagen HSV de OpenCV (H en 0-179, S y V en 0-255).
        mask: Máscara 0/255 del plátano.
        value_max: Umbral de V (0-255) bajo el cual un píxel es mancha
            (``features.dark_spot_value_max``).
        morph_kernel: Tamaño impar del elemento estructurante
            (``segmentation.morph_kernel``).

    Returns:
        ``SpotStats`` con el número de manchas y su área relativa; ceros si la
        máscara está vacía.

    Examples:
        >>> stats = count_dark_spots(hsv, seg.mask, 60, 7)
        >>> stats.count >= 0
        True
    """
    banana_pixels = int(cv2.countNonZero(mask))
    if banana_pixels == 0:
        return SpotStats(count=0, area_ratio=0.0)
    dark = ((hsv[:, :, 2] < value_max) & (mask == 255)).astype(np.uint8) * 255  # V en 0-255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    n_labels, _ = cv2.connectedComponents(dark)
    return SpotStats(
        count=n_labels - 1,  # la etiqueta 0 es el fondo
        area_ratio=float(cv2.countNonZero(dark)) / banana_pixels,
    )


def extract_row(path: Path, label: str, pipeline: BananaPipeline) -> dict[str, str | float | int]:
    """Calcula la fila del CSV de una imagen.

    Args:
        path: Ruta a la imagen.
        label: Etiqueta (nombre de carpeta de la clase).
        pipeline: Pipeline configurado.

    Returns:
        Diccionario con las claves de ``CSV_COLUMNS``.

    Raises:
        ImageLoadError: Si la imagen no se puede leer.
        SegmentationError: Si no se detecta un plátano.

    Examples:
        >>> row = extract_row(Path("data/train/inmaduro/a.jpg"), "inmaduro", pipe)
        >>> row["etiqueta"]
        'inmaduro'
    """
    cfg = pipeline.thresholds
    result = pipeline.run(load_image(path))
    fs = result.verdict.features
    spots = count_dark_spots(
        result.preprocessed.hsv,
        result.segmentation.mask,
        cfg.features.dark_spot_value_max,
        cfg.segmentation.morph_kernel,
    )
    return {
        PATH_COLUMN: path.as_posix(),
        LABEL_COLUMN: label,
        "pct_verde": fs.green_ratio,
        "pct_amarillo": fs.yellow_ratio,
        "pct_marron": fs.brown_ratio,
        "pct_oscuro": fs.dark_spot_ratio,
        "n_manchas": spots.count,
        "area_manchas_pct": spots.area_ratio,
        "other_ratio": fs.other_ratio,
    }


def extract_dataset(
    split_dir: Path, pipeline: BananaPipeline, logger: logging.Logger
) -> list[dict[str, str | float | int]]:
    """Extrae las características de todas las imágenes de una partición.

    Args:
        split_dir: Carpeta ``data/<split>`` con una subcarpeta por clase.
        pipeline: Pipeline configurado.
        logger: Logger donde se registran las imágenes que fallan.

    Returns:
        Una fila por imagen procesada con éxito, en orden determinista.

    Raises:
        DatasetError: Si la carpeta no existe, no contiene imágenes o todas fallaron.

    Examples:
        >>> rows = extract_dataset(Path("data/train"), pipe, logging.getLogger())
    """
    split = split_dir.name
    if not split_dir.is_dir():
        raise DatasetError(
            f"No existe la carpeta {split_dir.as_posix()}. Estructura esperada:\n"
            f"{expected_structure(split, split_dir.parent)}"
        )
    samples = list(iter_dataset(split_dir))
    if not samples:
        raise DatasetError(
            f"No hay imágenes en {split_dir.as_posix()}. Estructura esperada:\n"
            f"{expected_structure(split, split_dir.parent)}"
        )

    rows: list[dict[str, str | float | int]] = []
    for sample in samples:
        try:
            rows.append(extract_row(sample.path, sample.path.parent.name, pipeline))
        except BananaError as exc:
            logger.warning("FALLO %s: %s", sample.path.as_posix(), exc)
    logger.info("Procesadas %d de %d imágenes", len(rows), len(samples))
    if not rows:
        raise DatasetError(
            f"Ninguna imagen de {split_dir.as_posix()} pudo procesarse; revisa el log."
        )
    return rows


def save_csv(rows: list[dict[str, str | float | int]], path: Path) -> None:
    """Escribe las filas en un CSV con el encabezado ``CSV_COLUMNS``.

    Args:
        rows: Filas devueltas por :func:`extract_dataset`.
        path: Archivo de salida (se crean las carpetas necesarias).

    Examples:
        >>> save_csv(rows, Path("data/features_train.csv"))
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_logger(log_path: Path) -> logging.Logger:
    """Crea un logger que escribe en un archivo y en la consola.

    Args:
        log_path: Archivo de log (se sobrescribe en cada ejecución).

    Returns:
        Logger listo para usarse.

    Examples:
        >>> build_logger(Path("data/extract_train.log")).name
        'extract_dataset.extract_train.log'
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"extract_dataset.{log_path.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for old in list(logger.handlers):
        old.close()
        logger.removeHandler(old)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y lee los argumentos de línea de comandos.

    Args:
        argv: Lista de argumentos; ``None`` usa ``sys.argv``.

    Returns:
        Namespace con ``split``, ``data_dir`` (Path) y ``config`` (Path).

    Examples:
        >>> parse_args(["--split", "train"]).split
        'train'
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--split", required=True, choices=SPLITS, help="Partición a procesar")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Carpeta de datos")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_THRESHOLDS_PATH, help="YAML de umbrales"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del script.

    Args:
        argv: Argumentos de línea de comandos; ``None`` usa ``sys.argv``.

    Returns:
        Código de salida: 0 si todo salió bien, 1 si el dataset es inválido.

    Examples:
        >>> main(["--split", "train"])  # doctest: +SKIP
        0
    """
    args = parse_args(argv)
    logger = build_logger(args.data_dir / f"extract_{args.split}.log")
    try:
        pipeline = BananaPipeline.from_yaml(args.config)
        rows = extract_dataset(args.data_dir / args.split, pipeline, logger)
    except BananaError as exc:
        logger.error("%s", exc)
        return 1
    output = args.data_dir / f"features_{args.split}.csv"
    save_csv(rows, output)
    logger.info("Guardado %s (%d filas)", output.as_posix(), len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
