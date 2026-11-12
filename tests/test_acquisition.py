from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from banana.acquisition import iter_dataset, load_from_bytes, load_image
from banana.common import BGRImage, ImageLoadError, Ripeness

pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="Esqueleto: pendiente")


def test_load_image_roundtrip(tmp_path: Path, yellow_banana: BGRImage) -> None:
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), yellow_banana)
    img = load_image(path)
    assert img.shape == yellow_banana.shape
    assert img.dtype == yellow_banana.dtype


def test_load_image_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "no_existe.jpg")


def test_load_from_bytes(yellow_banana: BGRImage) -> None:
    ok, buf = cv2.imencode(".png", yellow_banana)
    assert ok
    assert load_from_bytes(buf.tobytes()).shape == yellow_banana.shape


def test_load_from_bytes_invalid() -> None:
    with pytest.raises(ImageLoadError):
        load_from_bytes(b"esto no es una imagen")


def test_iter_dataset_labels_from_folders(tmp_path: Path, green_banana: BGRImage) -> None:
    for folder in ("inmaduro", "comestible", "no_comestible", "otra_cosa"):
        (tmp_path / folder).mkdir()
        cv2.imwrite(str(tmp_path / folder / "a.jpg"), green_banana)
    samples = list(iter_dataset(tmp_path))
    assert {s.label for s in samples} == set(Ripeness)
    assert len(samples) == 3
