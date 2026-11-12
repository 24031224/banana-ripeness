from __future__ import annotations

from pathlib import Path

import pytest

from banana.classifier import (
    LABEL_COLUMN,
    TREE_FEATURES,
    ClassifierError,
    below_confidence,
    classify_rules,
    classify_tree,
    exceed_confidence,
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


def test_edible_banana_has_clear_confidence(thresholds: Thresholds) -> None:
    """Caso reportado: verde 0%, amarillo 63%, marrón 22%, manchas 2%."""
    v = classify_rules(_fs(green=0.0, yellow=0.63, brown=0.22, dark=0.02), thresholds.classifier)
    assert v.label is Ripeness.COMESTIBLE
    assert v.confidence > 0.7  # antes: 0.30
    # media geométrica de (0.45, 0.92, 1.0)
    assert v.confidence == pytest.approx((0.45 * 0.92 * 1.0) ** (1 / 3), abs=1e-3)


def test_edible_confidence_monotonic_in_each_distance(thresholds: Thresholds) -> None:
    base = classify_rules(_fs(brown=0.20, dark=0.10, green=0.10), thresholds.classifier)
    for closer in (_fs(brown=0.30, dark=0.10, green=0.10), _fs(brown=0.20, dark=0.20, green=0.10)):
        assert classify_rules(closer, thresholds.classifier).confidence < base.confidence
    closer_green = classify_rules(_fs(brown=0.20, dark=0.10, green=0.30), thresholds.classifier)
    assert closer_green.confidence < base.confidence


def test_edible_confidence_extremes(thresholds: Thresholds) -> None:
    assert classify_rules(_fs(yellow=1.0), thresholds.classifier).confidence == 1.0
    near = classify_rules(_fs(brown=0.399), thresholds.classifier)
    assert near.label is Ripeness.COMESTIBLE
    assert near.confidence < 0.2  # (0.0025 · 1 · 1)^(1/3) ≈ 0.136


def test_exceed_confidence_grows_with_margin(thresholds: Thresholds) -> None:
    cfg = thresholds.classifier
    values = [0.40, 0.5, 0.7, 1.0]
    confs = [classify_rules(_fs(brown=b), cfg).confidence for b in values]
    assert confs == sorted(confs)
    assert confs[0] == 0.0
    assert confs[-1] == 1.0
    green = [classify_rules(_fs(green=g), cfg).confidence for g in (0.35, 0.6, 1.0)]
    assert green == sorted(green)


@pytest.mark.parametrize("value", [-0.5, 0.0, 0.2, 0.4, 0.9, 1.0, 2.0])
@pytest.mark.parametrize("threshold", [0.0, 0.25, 0.4, 1.0])
def test_confidence_helpers_in_unit_interval(value: float, threshold: float) -> None:
    assert 0.0 <= exceed_confidence(value, threshold) <= 1.0
    assert 0.0 <= below_confidence(value, threshold) <= 1.0


def test_exceed_and_below_values() -> None:
    assert exceed_confidence(0.70, 0.40) == pytest.approx(0.5)
    assert exceed_confidence(0.30, 0.40) == 0.0
    assert below_confidence(0.22, 0.40) == pytest.approx(0.45)
    assert below_confidence(0.0, 0.40) == 1.0
    assert below_confidence(0.40, 0.40) == 0.0


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_exceed_below_reject_bad_threshold(bad: float) -> None:
    with pytest.raises(ValueError):
        exceed_confidence(0.5, bad)
    with pytest.raises(ValueError):
        below_confidence(0.5, bad)


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
