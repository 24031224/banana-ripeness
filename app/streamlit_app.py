"""Interfaz web de demostración: sube una foto y obtén la madurez del plátano.

Ejecutar con::

    streamlit run app/streamlit_app.py

La app solo conoce :class:`banana.pipeline.BananaPipeline` y los tipos de
:mod:`banana.common`; no llama directamente a las etapas internas.
"""

from __future__ import annotations

from banana.common import BGRImage, PipelineResult
from banana.pipeline import BananaPipeline


def get_pipeline() -> BananaPipeline:
    """Crea (una sola vez por sesión, con ``st.cache_resource``) el pipeline desde el YAML.

    Returns:
        Pipeline configurado con ``config/thresholds.yaml``.

    Examples:
        >>> pipe = get_pipeline()
    """
    raise NotImplementedError


def read_input_image() -> BGRImage | None:
    """Muestra el selector de archivo / cámara y devuelve la imagen elegida.

    Returns:
        Imagen BGR, o ``None`` si el usuario aún no ha subido nada.

    Examples:
        >>> img = read_input_image()
    """
    raise NotImplementedError


def render_result(result: PipelineResult) -> None:
    """Dibuja el veredicto, la confianza, las razones y las imágenes intermedias.

    Recordatorio: Streamlit espera RGB; convertir con ``cv2.cvtColor(..., COLOR_BGR2RGB)``.

    Args:
        result: Resultado completo del pipeline.

    Examples:
        >>> render_result(pipe.run(img))
    """
    raise NotImplementedError


def main() -> None:
    """Punto de entrada de la app: configura la página y orquesta la interacción.

    Debe capturar ``SegmentationError`` e ``ImageLoadError`` y mostrarlos con ``st.error``.

    Examples:
        >>> main()
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
