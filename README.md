# banana-ripeness

Clasificador de madurez de plátanos con **visión por computadora clásica** (OpenCV):
espacio de color HSV, segmentación por color + morfología y reglas explicables.

| Clase | Idea de la regla |
|---|---|
| **Inmaduro** | Predomina el verde |
| **Comestible** | Predomina el amarillo, pocas manchas |
| **No comestible** | Mucho café o muchas manchas oscuras |

> Estado: **esqueleto**. Las funciones tienen firma y docstring pero lanzan
> `NotImplementedError`.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Uso

```bash
streamlit run app/streamlit_app.py                 # demo web
python scripts/calibrate.py --dataset data/raw     # proponer umbrales
python scripts/evaluate.py  --dataset data/test    # métricas
pytest                                             # pruebas
mkdocs serve                                       # documentación
```

## Estructura

```
src/banana/          paquete: common, config, acquisition, preprocess,
                     segmentation, features, classifier, pipeline
app/                 interfaz Streamlit
scripts/             calibración y evaluación
config/              thresholds.yaml (todos los umbrales)
tests/               pytest (imágenes sintéticas en conftest.py)
docs/                MkDocs Material + mkdocstrings
data/samples/        muestras pequeñas versionadas (el resto de data/ se ignora)
```

## Contribuir

Lee [CLAUDE.md](CLAUDE.md) (idéntico a [AGENTS.md](AGENTS.md)) y
[docs/arquitectura.md](docs/arquitectura.md). Commits: `feat(modulo): descripción`.
