"""Interfaz web de demostración: sube una foto y obtén la madurez del plátano.

Ejecutar con::

    streamlit run app/streamlit_app.py

La app solo orquesta la interfaz: todo el procesamiento y la clasificación los hace
:class:`banana.pipeline.BananaPipeline`. Los sliders de "Avanzado" sobrescriben en
memoria copias de los umbrales (``dataclasses.replace``) que se pasan al pipeline;
el YAML nunca se modifica.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import streamlit as st

from banana.common import BananaError, BGRImage, Mask, PipelineResult, Ripeness
from banana.config import Thresholds
from banana.pipeline import BananaPipeline

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
_APP_DIR: Final[Path] = Path(__file__).resolve().parent
_SAMPLES_DIR: Final[Path] = _APP_DIR.parent / "data" / "samples"
_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
_NO_SELECTION: Final[str] = "— Seleccionar —"

_CONTOUR_COLOR_BGR: Final[tuple[int, int, int]] = (255, 0, 255)  # magenta (BGR)
_CONTOUR_THICKNESS: Final[int] = 2

_LABEL_COLORS: Final[dict[Ripeness, str]] = {
    Ripeness.INMADURO: "#2ECC40",
    Ripeness.COMESTIBLE: "#FFDC00",
    Ripeness.NO_COMESTIBLE: "#FF4136",
}

_LABEL_EMOJIS: Final[dict[Ripeness, str]] = {
    Ripeness.INMADURO: "🟢",
    Ripeness.COMESTIBLE: "🟡",
    Ripeness.NO_COMESTIBLE: "🔴",
}


# ---------------------------------------------------------------------------
# Configuración y ejecución
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_default_thresholds() -> Thresholds:
    """Carga los umbrales del YAML una sola vez por proceso.

    Returns:
        Umbrales de ``config/thresholds.yaml`` (inmutables).

    Raises:
        ConfigError: Si el YAML es inválido.

    Examples:
        >>> _load_default_thresholds().preprocess.target_width
        640
    """
    return BananaPipeline.from_yaml().thresholds


def _apply_overrides(
    base: Thresholds,
    *,
    brown_min: float,
    dark_min: float,
    green_min: float,
    dark_value_max: int,
    min_area_ratio: float,
) -> Thresholds:
    """Devuelve una copia de ``base`` con los valores de los sliders sobrescritos.

    Args:
        base: Umbrales originales (no se modifican).
        brown_min: ``no_comestible_brown_ratio_min``.
        dark_min: ``no_comestible_dark_ratio_min``.
        green_min: ``inmaduro_green_ratio_min``.
        dark_value_max: ``dark_spot_value_max`` (V en 0-255).
        min_area_ratio: ``min_area_ratio`` de la segmentación, en [0, 1].

    Returns:
        Nuevo ``Thresholds`` con los cambios aplicados.

    Examples:
        >>> th = _apply_overrides(
        ...     base, brown_min=0.5, dark_min=0.3, green_min=0.4,
        ...     dark_value_max=50, min_area_ratio=0.03,
        ... )
    """
    return replace(
        base,
        segmentation=replace(base.segmentation, min_area_ratio=min_area_ratio),
        features=replace(base.features, dark_spot_value_max=dark_value_max),
        classifier=replace(
            base.classifier,
            no_comestible_brown_ratio_min=brown_min,
            no_comestible_dark_ratio_min=dark_min,
            inmaduro_green_ratio_min=green_min,
        ),
    )


# ---------------------------------------------------------------------------
# Helpers de visualización
# ---------------------------------------------------------------------------
def _bgr_to_rgb(image: BGRImage) -> np.ndarray:
    """Convierte BGR a RGB para Streamlit.

    Args:
        image: Imagen BGR.

    Returns:
        Imagen RGB.

    Examples:
        >>> rgb = _bgr_to_rgb(bgr)
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _draw_mask_contour(bgr: BGRImage, mask: Mask) -> np.ndarray:
    """Dibuja el contorno magenta de la máscara sin teñir el interior.

    Args:
        bgr: Imagen BGR sobre la que dibujar (no se modifica).
        mask: Máscara 0/255 del plátano.

    Returns:
        Imagen RGB con el contorno magenta.

    Examples:
        >>> rgb = _draw_mask_contour(result.preprocessed.bgr, result.segmentation.mask)
    """
    canvas = bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, _CONTOUR_COLOR_BGR, _CONTOUR_THICKNESS)
    return _bgr_to_rgb(canvas)


def _ratio_bar(name: str, ratio: float, bar_color: str, border: str = "none") -> None:
    """Dibuja una barra horizontal de porcentaje.

    Args:
        name: Texto de la etiqueta (con emoji).
        ratio: Proporción en [0, 1].
        bar_color: Color CSS de la barra.
        border: Borde CSS opcional de la barra.

    Examples:
        >>> _ratio_bar("🟢 Verde", 0.8, "#2ECC40")
    """
    pct = ratio * 100
    st.markdown(
        f"""
        <div style="margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                <span>{name}</span>
                <span><b>{pct:.1f}%</b></span>
            </div>
            <div style="background: #333; border-radius: 6px; overflow: hidden; height: 22px;">
                <div style="width: {pct:.1f}%; background: {bar_color}; height: 100%;
                    border-radius: 6px 0 0 6px; border: {border};
                    transition: width 0.4s ease;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Renderizado principal
# ---------------------------------------------------------------------------
def render_result(result: PipelineResult) -> None:
    """Dibuja el veredicto, imágenes, barras de color y reglas disparadas.

    Args:
        result: Resultado completo del pipeline.

    Examples:
        >>> render_result(pipeline.run(img))
    """
    verdict = result.verdict
    features = verdict.features
    color = _LABEL_COLORS[verdict.label]
    emoji = _LABEL_EMOJIS[verdict.label]

    st.markdown(
        f"""
        <div style="text-align: center; padding: 1.2rem 0 0.6rem 0;">
            <span style="font-size: 3rem;">{emoji}</span>
            <h1 style="color: {color}; margin: 0.2rem 0 0; font-size: 2.6rem;
                text-shadow: 1px 1px 4px rgba(0,0,0,0.3);">{verdict.label.value}</h1>
            <p style="color: #888; font-size: 1rem; margin-top: 0.3rem;">
                Confianza: {verdict.confidence:.0%}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bgr = result.preprocessed.bgr
    col_orig, col_mask = st.columns(2)
    with col_orig:
        st.image(_bgr_to_rgb(bgr), caption="Imagen original", width="stretch")
    with col_mask:
        st.image(
            _draw_mask_contour(bgr, result.segmentation.mask),
            caption="Máscara (contorno)",
            width="stretch",
        )

    st.markdown("#### 🎨 Distribución de color en el plátano")
    _ratio_bar("🟢 Verde", features.green_ratio, "#2ECC40")
    _ratio_bar("🟡 Amarillo", features.yellow_ratio, "#FFDC00")
    _ratio_bar("🟤 Marrón", features.brown_ratio, "#8B4513")
    _ratio_bar("⚪ Otros", features.other_ratio, "#9E9E9E")

    with st.expander("❓ ¿Por qué?"):
        st.markdown("**Reglas disparadas:**")
        for reason in verdict.reasons:
            st.markdown(f"- `{reason}`")
        st.markdown("---")
        st.markdown(
            "ℹ️ El clasificador evalúa reglas en orden de prioridad: "
            "primero comprueba si el plátano es **No comestible** (exceso de marrón "
            "o manchas oscuras), luego si es **Inmaduro** (predominio de verde). "
            "Si ninguna regla aplica, se clasifica como **Comestible**."
        )


# ---------------------------------------------------------------------------
# Sidebar: entrada de imagen + sliders avanzados
# ---------------------------------------------------------------------------
def sidebar_input(base: Thresholds) -> tuple[BGRImage | None, Thresholds]:
    """Muestra el sidebar con entrada de imagen y sliders de umbrales.

    Args:
        base: Umbrales por defecto (se usan como valores iniciales de los sliders).

    Returns:
        Tupla ``(imagen_bgr_o_None, umbrales_con_overrides)``.

    Examples:
        >>> image, thresholds = sidebar_input(_load_default_thresholds())
    """
    with st.sidebar:
        st.markdown("## 🍌 Entrada")
        uploaded = st.file_uploader(
            "Sube una imagen",
            type=[s.lstrip(".") for s in sorted(_IMAGE_SUFFIXES)],
            help="Arrastra una foto de un plátano aquí",
        )
        camera = st.camera_input("O toma una foto con la cámara")
        st.markdown("---")

        sample_files = sorted(
            p for p in _SAMPLES_DIR.glob("*") if p.suffix.lower() in _IMAGE_SUFFIXES
        )
        choice = _NO_SELECTION
        if sample_files:
            st.markdown("#### 📂 Imágenes de ejemplo")
            choice = st.selectbox(
                "Elige una muestra", [_NO_SELECTION] + [p.name for p in sample_files], index=0
            )
        st.markdown("---")

        with st.expander("⚙️ Avanzado"):
            st.caption("Sobrescribe en memoria los umbrales del YAML (no se guardan).")
            clf = base.classifier
            brown_min = st.slider(
                "Umbral marrón → No comestible",
                0.0,
                1.0,
                clf.no_comestible_brown_ratio_min,
                0.01,
                help="brown_ratio mínimo para clasificar como No comestible",
            )
            dark_min = st.slider(
                "Umbral manchas → No comestible",
                0.0,
                1.0,
                clf.no_comestible_dark_ratio_min,
                0.01,
                help="dark_spot_ratio mínimo para No comestible",
            )
            green_min = st.slider(
                "Umbral verde → Inmaduro",
                0.0,
                1.0,
                clf.inmaduro_green_ratio_min,
                0.01,
                help="green_ratio mínimo para clasificar como Inmaduro",
            )
            dark_value_max = st.slider(
                "V máximo para manchas oscuras",
                0,
                255,
                base.features.dark_spot_value_max,
                1,
                help="Píxeles con V < este valor cuentan como mancha (0-255)",
            )
            min_area = st.slider(
                "Área mínima del plátano",
                0.0,
                0.5,
                base.segmentation.min_area_ratio,
                0.005,
                format="%.3f",
                help="Fracción mínima de la imagen que debe ocupar el plátano",
            )

    thresholds = _apply_overrides(
        base,
        brown_min=brown_min,
        dark_min=dark_min,
        green_min=green_min,
        dark_value_max=dark_value_max,
        min_area_ratio=min_area,
    )

    # Prioridad: subida > cámara > ejemplo
    image_bytes: bytes | None = None
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
    elif camera is not None:
        image_bytes = camera.getvalue()
    elif choice != _NO_SELECTION and (_SAMPLES_DIR / choice).exists():
        image_bytes = (_SAMPLES_DIR / choice).read_bytes()

    if image_bytes is None:
        return None, thresholds

    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        st.sidebar.error("⚠️ No se pudo decodificar la imagen.")
        return None, thresholds
    return decoded, thresholds


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
def main() -> None:
    """Punto de entrada de la app: configura la página y orquesta la interacción.

    Los errores del dominio (``BananaError``) se muestran con ``st.error``.

    Examples:
        >>> main()
    """
    st.set_page_config(page_title="🍌 Clasificador de Plátanos", page_icon="🍌", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stImage"] img {
            border-radius: 8px;
            border: 1px solid rgba(128,128,128,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🍌 Clasificador de Madurez de Plátanos")
    st.caption("Visión por computadora clásica · OpenCV + reglas HSV")

    try:
        base = _load_default_thresholds()
    except BananaError as exc:
        st.error(f"⚠️ No se pudo cargar la configuración: {exc}")
        return

    image, thresholds = sidebar_input(base)
    if image is None:
        st.info(
            "👈 Sube una imagen, toma una foto o elige un ejemplo desde la barra lateral "
            "para comenzar el análisis."
        )
        return

    try:
        result = BananaPipeline(thresholds).run(image)
    except BananaError as exc:
        st.error(f"⚠️ {exc}")
        st.image(_bgr_to_rgb(image), caption="Imagen proporcionada", width="stretch")
        return

    render_result(result)


if __name__ == "__main__":
    main()
