# Reglas del proyecto: clasificador de madurez de plátanos

Clasificador con **OpenCV clásico (HSV + reglas)**, sin deep learning como método principal.
Clases: `Inmaduro`, `Comestible`, `No comestible`. Equipo de 5 personas.

## Reglas obligatorias

1. **Python 3.11** y **type hints en todo** (funciones, argumentos, retornos; también en tests).
2. **Docstrings estilo Google en español.** Los encabezados de sección se dejan en inglés
   porque son palabras clave del parser (`Args:`, `Returns:`, `Raises:`, `Yields:`,
   `Attributes:`, `Examples:`); el contenido va en español.
3. **Cada función pública lleva docstring** con: qué hace, args, returns y un ejemplo corto.
4. **Umbrales SIEMPRE en `config/thresholds.yaml`**, nunca hardcodeados. Cada etapa recibe
   su dataclass de config (`PreprocessConfig`, `SegmentationConfig`, ...) como argumento.
5. **Cada módulo nuevo lleva su test** en `tests/test_<modulo>.py` **y una página** en
   `docs/modulos/<modulo>.md` (con `::: banana.<modulo>`) registrada en `mkdocs.yml`.
6. **OpenCV usa H en 0-179** (grados / 2), S y V en 0-255. Recuérdalo en comentarios cada
   vez que aparezca un valor de tono.
7. **Commits en español**, formato `tipo(modulo): descripción`, p. ej.
   `feat(segmentation): agrega limpieza morfológica`.
   Tipos: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`.

## Arquitectura

```
src/banana/
  common.py        # CONTRATO: tipos compartidos, enums, excepciones
  config.py        # dataclasses de config + load_thresholds()
  acquisition.py   # archivo / bytes / cámara / dataset → BGRImage
  preprocess.py    # BGRImage → PreprocessedImage (bgr + hsv)
  segmentation.py  # HSVImage → Segmentation (máscara 0/255)
  features.py      # HSVImage + Segmentation → FeatureSet
  classifier.py    # FeatureSet → Verdict (reglas explicables)
  pipeline.py      # BananaPipeline: orquesta todo → PipelineResult
app/streamlit_app.py    # solo usa BananaPipeline
scripts/calibrate.py    # propone umbrales desde un dataset etiquetado
scripts/evaluate.py     # métricas y matriz de confusión
```

Invariantes: imágenes en BGR `uint8`; máscaras `uint8` con 0/255; proporciones en [0, 1]
relativas al plátano; dataclasses `frozen`; errores heredan de `BananaError`.
`common.py` y `config.py` solo se cambian con un PR acordado por el equipo.
Detalle completo en `docs/arquitectura.md`.

## Comandos

```bash
pip install -r requirements.txt     # instala también el paquete en modo editable
ruff check . && ruff format --check .
pytest
streamlit run app/streamlit_app.py
mkdocs serve
```

## Tests del esqueleto

Los tests de módulos sin implementar están marcados con
`pytest.mark.xfail(raises=NotImplementedError)`. Al implementar un módulo, quita la marca:
sus tests pasan a ser obligatorios. Los datos de prueba son imágenes sintéticas generadas
en `tests/conftest.py`; la config de prueba se construye a mano allí.

## Datos

`data/` está ignorado por git excepto `data/samples/` (pocas imágenes pequeñas).
Dataset: `data/<split>/{inmaduro,comestible,no_comestible}/*.jpg`.
