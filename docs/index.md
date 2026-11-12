# Clasificador de madurez de plátanos

Proyecto académico de IA que clasifica un plátano en **Inmaduro**, **Comestible** o
**No comestible** usando visión por computadora clásica: espacio de color HSV,
segmentación por color + morfología y un clasificador de reglas explicables.
No se usa deep learning como método principal.

## Inicio rápido

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pytest
streamlit run app/streamlit_app.py
mkdocs serve                     # esta documentación en http://127.0.0.1:8000
```

## Mapa de la documentación

- [Arquitectura y contrato](arquitectura.md): cómo se conectan los módulos.
- [Configuración](configuracion.md): qué significa cada umbral.
- [Cómo contribuir](contribuir.md): reglas del proyecto y flujo de trabajo.
- *Referencia de módulos*: generada automáticamente desde los docstrings.
