from __future__ import annotations

from pathlib import Path

import pytest

from banana.classifier import (
    LABEL_COLUMN,
    TREE_FEATURES,
    ClassifierError,
    classify_rules,
    classify_tree,
    export_tree_rules,
    feature_vector,
    margin_confidence,
    train_tree,
)
from banana.common import FeatureSet, Ripeness
from banana.config import Thresholds


def _fs(
    green: float = 0.0, yellow: float = 0.0, brown: float = 0.0, dark: float = 0.0
) -> FeatureSet:
    return FeatureSet(
        hue_mean=27.0,
        hue_std=5.0,
        saturation_mean=180.0,
        value_mean=200.0,
        green_ratio=green,
        yellow_ratio=yellow,
        brown_ratio=brown,
        dark_spot_ratio=dark,
        hue_histogram=(1.0,),
    )


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (_fs(green=0.8, yellow=0.2), Ripeness.INMADURO),
        (_fs(yellow=0.9, brown=0.1), Ripeness.COMESTIBLE),
        (_fs(yellow=0.3, brown=0.7), Ripeness.NO_COMESTIBLE),
        (_fs(yellow=0.6, dark=0.3), Ripeness.NO_COMESTIBLE),
        # Prioridad: "No comestible" gana aunque también haya mucho verde
        (_fs(green=0.5, brown=0.5), Ripeness.NO_COMESTIBLE),
        # Justo en el umbral la regla se cumple (>=)
        (_fs(brown=0.40), Ripeness.NO_COMESTIBLE),
        (_fs(dark=0.25), Ripeness.NO_COMESTIBLE),
        (_fs(green=0.35), Ripeness.INMADURO),
        # Justo por debajo de todos los umbrales
        (_fs(green=0.34, brown=0.39, dark=0.24), Ripeness.COMESTIBLE),
    ],
)
def test_rules(features: FeatureSet, expected: Ripeness, thresholds: Thresholds) -> None:
    v = classify_rules(features, thresholds.classifier)
    assert v.label is expected
    assert 0.0 <= v.confidence <= 1.0
    assert v.reasons
    assert v.features is features


@pytest.mark.parametrize(
    ("features", "expected_reasons"),
    [
        (_fs(brown=0.52), ("brown_ratio=0.52 >= 0.40",)),
        (_fs(dark=0.30), ("dark_spot_ratio=0.30 >= 0.25",)),
        (
            _fs(brown=0.50, dark=0.30),
            ("brown_ratio=0.50 >= 0.40", "dark_spot_ratio=0.30 >= 0.25"),
        ),
        (_fs(green=0.80), ("green_ratio=0.80 >= 0.35",)),
        (
            _fs(yellow=0.9),
            (
                "brown_ratio=0.00 < 0.40",
                "dark_spot_ratio=0.00 < 0.25",
                "green_ratio=0.00 < 0.35",
            ),
        ),
    ],
)
def test_rules_explain_decision(
    features: FeatureSet, expected_reasons: tuple[str, ...], thresholds: Thresholds
) -> None:
    assert classify_rules(features, thresholds.classifier).reasons == expected_reasons


def test_rules_confidence_grows_with_margin(thresholds: Thresholds) -> None:
    cerca = classify_rules(_fs(brown=0.41), thresholds.classifier)
    lejos = classify_rules(_fs(brown=0.95), thresholds.classifier)
    assert cerca.confidence < lejos.confidence


def test_confidence_zero_at_threshold() -> None:
    assert margin_confidence(0.4, 0.4) == 0.0


def test_confidence_saturates() -> None:
    assert margin_confidence(1.0, 0.1) <= 1.0


def test_confidence_symmetric() -> None:
    assert margin_confidence(0.5, 0.4) == pytest.approx(margin_confidence(0.3, 0.4))


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_confidence_rejects_bad_threshold(bad: float) -> None:
    with pytest.raises(ValueError):
        margin_confidence(0.5, bad)


# --------------------------------------------------------------------------- #
# Árbol de decisión
# --------------------------------------------------------------------------- #
_TREE_SAMPLES: list[tuple[FeatureSet, str]] = [
    (_fs(green=0.80, yellow=0.20), "inmaduro"),
    (_fs(green=0.70, yellow=0.30), "inmaduro"),
    (_fs(green=0.65, yellow=0.30, brown=0.05), "Inmaduro"),
    (_fs(yellow=0.90, brown=0.10), "comestible"),
    (_fs(yellow=0.80, brown=0.15), "comestible"),
    (_fs(green=0.05, yellow=0.85, brown=0.10), "Comestible"),
    (_fs(yellow=0.30, brown=0.70, dark=0.30), "no_comestible"),
    (_fs(yellow=0.20, brown=0.75, dark=0.35), "no_comestible"),
    (_fs(yellow=0.40, brown=0.60, dark=0.20), "No comestible"),
]


def _write_csv(path: Path, rows: list[tuple[FeatureSet, str]]) -> Path:
    lines = [",".join((*TREE_FEATURES, LABEL_COLUMN))]
    for fs, label in rows:
        lines.append(",".join([*(str(v) for v in feature_vector(fs)), label]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def tree_csv(tmp_path: Path) -> Path:
    return _write_csv(tmp_path / "features.csv", _TREE_SAMPLES)


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (_fs(green=0.75, yellow=0.25), Ripeness.INMADURO),
        (_fs(yellow=0.85, brown=0.10), Ripeness.COMESTIBLE),
        (_fs(yellow=0.25, brown=0.72, dark=0.30), Ripeness.NO_COMESTIBLE),
    ],
)
def test_tree_classifies_each_class(
    features: FeatureSet, expected: Ripeness, tree_csv: Path
) -> None:
    v = classify_tree(train_tree(tree_csv), features)
    assert v.label is expected
    assert 0.0 <= v.confidence <= 1.0
    assert v.reasons
    assert v.features is features


def test_tree_depth_is_limited(tree_csv: Path) -> None:
    assert train_tree(tree_csv).get_depth() <= 3


def test_tree_export_mentions_features_and_classes(tree_csv: Path) -> None:
    text = export_tree_rules(train_tree(tree_csv))
    assert "class:" in text
    assert any(name in text for name in TREE_FEATURES)
    assert "Comestible" in text


def test_tree_reasons_describe_path(tree_csv: Path) -> None:
    v = classify_tree(train_tree(tree_csv), _fs(yellow=0.25, brown=0.72, dark=0.30))
    assert all(("<=" in r or ">" in r) for r in v.reasons)


def test_train_tree_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("hue_mean,label\n20,inmaduro\n", encoding="utf-8")
    with pytest.raises(ClassifierError, match="Faltan columnas"):
        train_tree(path)


def test_train_tree_unknown_label(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "x.csv", [(_fs(green=0.8), "verde"), (_fs(brown=0.8), "no_comestible")]
    )
    with pytest.raises(ClassifierError, match="Etiqueta desconocida"):
        train_tree(path)


def test_train_tree_needs_two_classes(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "one.csv", [(_fs(green=0.8), "inmaduro"), (_fs(green=0.7), "inmaduro")]
    )
    with pytest.raises(ClassifierError, match="dos clases"):
        train_tree(path)


def test_train_tree_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ClassifierError):
        train_tree(tmp_path / "no_existe.csv")
