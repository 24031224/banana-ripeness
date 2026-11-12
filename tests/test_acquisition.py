from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from banana.acquisition import capture_from_camera, iter_dataset, load_from_bytes, load_image
from banana.common import BGRImage, ImageLoadError, Ripeness


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


def test_load_image_path_with_accents_and_spaces(tmp_path: Path, yellow_banana: BGRImage) -> None:
    folder = tmp_path / "carpeta con ñ y tildes áé"
    folder.mkdir()
    path = folder / "plátano maduro.png"
    ok, buf = cv2.imencode(".png", yellow_banana)
    assert ok
    buf.tofile(path)
    assert load_image(str(path)).shape == yellow_banana.shape


def test_load_image_invalid_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "falso.jpg"
    path.write_bytes(b"no soy un jpg")
    with pytest.raises(ImageLoadError):
        load_image(path)


def test_load_image_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "vacio.jpg"
    path.write_bytes(b"")
    with pytest.raises(ImageLoadError):
        load_image(path)


def test_load_from_bytes_empty() -> None:
    with pytest.raises(ImageLoadError):
        load_from_bytes(b"")


def test_iter_dataset_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_dataset(tmp_path / "no_existe"))


def test_iter_dataset_filters_extensions_and_is_sorted(
    tmp_path: Path, green_banana: BGRImage
) -> None:
    folder = tmp_path / "comestible"
    folder.mkdir()
    cv2.imwrite(str(folder / "b.JPG"), green_banana)
    cv2.imwrite(str(folder / "a.png"), green_banana)
    (folder / "notas.txt").write_text("x", encoding="utf-8")
    samples = list(iter_dataset(tmp_path))
    assert [s.path.name for s in samples] == ["a.png", "b.JPG"]
    assert all(s.label is Ripeness.COMESTIBLE for s in samples)


class _ClosedCapture:
    def __init__(self, index: int) -> None:
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return False

    def release(self) -> None:
        self.released = True


def test_capture_from_camera_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", _ClosedCapture)
    with pytest.raises(ImageLoadError):
        capture_from_camera(0)


def test_capture_from_camera_returns_frame(
    monkeypatch: pytest.MonkeyPatch, yellow_banana: BGRImage
) -> None:
    class OpenCapture:
        def __init__(self, index: int) -> None:
            pass

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def read(self) -> tuple[bool, BGRImage]:
            return True, yellow_banana

        def release(self) -> None:
            pass

    monkeypatch.setattr(cv2, "VideoCapture", OpenCapture)
    assert capture_from_camera(0).shape == yellow_banana.shape
