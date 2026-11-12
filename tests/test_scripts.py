"""Tests de scripts/extract_dataset.py y scripts/train_tree.py con un mini-dataset sintético."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from banana.common import BGRImage
from banana.config import Thresholds
from banana.pipeline import BananaPipeline
from conftest import HSV_BROWN, HSV_GREEN, HSV_YELLOW, make_banana_image
from extract_dataset import (
    CSV_COLUMNS,
    DatasetError,
    count_dark_spots,
    extract_dataset,
    save_csv,
)
from extract_dataset import main as extract_main
from train_tree import (
    CRITICAL_CLASS,
    RESULT_FILES,
    TREE_RANDOM_STATE,
    Dataset,
    compute_metrics,
    fit_tree,
    load_features_csv,
    predict_rules,
)
from train_tree import main as train_main

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"
CLASS_COLORS: dict[str, tuple[int, int, int]] = {  # HSV con H en 0-179
    "inmaduro": HSV_GREEN,
    "comestible": HSV_YELLOW,
    "no_comestible": HSV_BROWN,
}
PER_CLASS = 3


def _write_image(path: Path, image: BGRImage) -> None:
    """Escribe un PNG (imencode + tofile: seguro con rutas de Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    buf.tofile(path)


def _make_split(root: Path, split: str) -> None:
    for folder, (h, s, v) in CLASS_COLORS.items():
        for i in range(PER_CLASS):
            _write_image(root / split / folder / f"{i}.png", make_banana_image((h + i, s, v)))


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    for split in ("train", "test"):
        _make_split(tmp_path / "data", split)
    return tmp_path / "data"


@pytest.fixture
def pipeline(thresholds: Thresholds) -> BananaPipeline:
    return BananaPipeline(thresholds)


def _make_csvs(dataset: Path, pipeline: BananaPipeline) -> tuple[Path, Path]:
    logger = logging.getLogger("tests.csv")
    paths: list[Path] = []
    for split in ("train", "test"):
        path = dataset / f"features_{split}.csv"
        save_csv(extract_dataset(dataset / split, pipeline, logger), path)
        paths.append(path)
    return paths[0], paths[1]


def test_extract_dataset_rows(dataset: Path, pipeline: BananaPipeline) -> None:
    rows = extract_dataset(dataset / "train", pipeline, logging.getLogger("tests.rows"))
    assert len(rows) == 3 * PER_CLASS
    by_label = {r["etiqueta"]: r for r in rows}
    assert float(by_label["inmaduro"]["pct_verde"]) > 0.9
    assert float(by_label["comestible"]["pct_amarillo"]) > 0.9
    assert float(by_label["no_comestible"]["pct_marron"]) > 0.9


def test_extract_logs_failures_and_continues(
    dataset: Path,
    pipeline: BananaPipeline,
    empty_image: BGRImage,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_image(dataset / "train" / "comestible" / "vacia.png", empty_image)
    with caplog.at_level(logging.WARNING):
        rows = extract_dataset(dataset / "train", pipeline, logging.getLogger("tests.fail"))
    assert len(rows) == 3 * PER_CLASS
    assert any("vacia.png" in m for m in caplog.messages)


def test_extract_missing_dir_shows_structure(tmp_path: Path, pipeline: BananaPipeline) -> None:
    with pytest.raises(DatasetError, match="no_comestible"):
        extract_dataset(tmp_path / "train", pipeline, logging.getLogger("tests.missing"))


def test_extract_empty_dir_shows_structure(tmp_path: Path, pipeline: BananaPipeline) -> None:
    (tmp_path / "train").mkdir()
    with pytest.raises(DatasetError, match="Estructura esperada"):
        extract_dataset(tmp_path / "train", pipeline, logging.getLogger("tests.empty"))


def test_count_dark_spots() -> None:
    hsv = np.full((60, 60, 3), (27, 200, 230), dtype=np.uint8)  # H en 0-179
    mask = np.full((60, 60), 255, dtype=np.uint8)
    cv2.circle(hsv, (15, 15), 6, (27, 200, 20), -1)  # V=20, bajo el umbral de prueba
    cv2.circle(hsv, (45, 45), 6, (27, 200, 20), -1)
    stats = count_dark_spots(hsv, mask, value_max=60, morph_kernel=3)
    assert stats.count == 2
    assert 0.0 < stats.area_ratio < 0.2
    empty = count_dark_spots(hsv, np.zeros_like(mask), value_max=60, morph_kernel=3)
    assert (empty.count, empty.area_ratio) == (0, 0.0)


def test_main_extract_writes_csv(dataset: Path) -> None:
    code = extract_main(
        ["--split", "train", "--data-dir", str(dataset), "--config", str(CONFIG_PATH)]
    )
    assert code == 0
    with (dataset / "features_train.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
        assert len(list(reader)) == 3 * PER_CLASS
    assert (dataset / "extract_train.log").is_file()


def test_main_extract_without_data_fails(tmp_path: Path) -> None:
    assert extract_main(["--split", "train", "--data-dir", str(tmp_path)]) == 1


def test_tree_and_rules_on_synthetic(dataset: Path, pipeline: BananaPipeline) -> None:
    train_csv, test_csv = _make_csvs(dataset, pipeline)
    train, test = load_features_csv(train_csv), load_features_csv(test_csv)
    tree = fit_tree(train)
    assert tree.get_depth() <= 3
    assert tree.random_state == TREE_RANDOM_STATE
    tree_m = compute_metrics(test.y, [str(p) for p in tree.predict(test.x)])
    rules_m = compute_metrics(test.y, predict_rules(test.x, pipeline.thresholds.classifier))
    assert tree_m.accuracy == 1.0
    assert rules_m.accuracy == 1.0
    assert rules_m.per_class[CRITICAL_CLASS]["recall"] == 1.0
    assert tree_m.confusion.tolist() == [[PER_CLASS, 0, 0], [0, PER_CLASS, 0], [0, 0, PER_CLASS]]


def test_main_train_writes_results(dataset: Path, pipeline: BananaPipeline, tmp_path: Path) -> None:
    train_csv, test_csv = _make_csvs(dataset, pipeline)
    out = tmp_path / "results"
    code = train_main(
        [
            "--train-csv",
            str(train_csv),
            "--test-csv",
            str(test_csv),
            "--out-dir",
            str(out),
            "--config",
            str(CONFIG_PATH),
        ]
    )
    assert code == 0
    for name in RESULT_FILES.values():
        assert (out / name).stat().st_size > 0
    comparison = (out / RESULT_FILES["comparison"]).read_text(encoding="utf-8")
    assert "Recall de `no_comestible`" in comparison
    assert "Reglas" in comparison


def test_main_train_without_csv_fails(tmp_path: Path) -> None:
    code = train_main(
        [
            "--train-csv",
            str(tmp_path / "a.csv"),
            "--test-csv",
            str(tmp_path / "b.csv"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1


def test_load_csv_requires_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("ruta,etiqueta,pct_verde\nx.png,otra,0.5\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="Faltan columnas"):
        load_features_csv(path)


def test_fit_tree_needs_two_classes(dataset: Path, pipeline: BananaPipeline) -> None:
    train_csv, _ = _make_csvs(dataset, pipeline)
    train = load_features_csv(train_csv)
    one_class = Dataset(x=train.x[:PER_CLASS], y=train.y[:PER_CLASS])
    with pytest.raises(DatasetError):
        fit_tree(one_class)
