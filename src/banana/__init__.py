"""Clasificador de madurez de plátanos con visión por computadora clásica.

El paquete implementa un pipeline determinista basado en OpenCV:
adquisición → preprocesamiento → segmentación → características → reglas.

Los tipos compartidos viven en :mod:`banana.common` y los umbrales en
``config/thresholds.yaml`` (cargados con :mod:`banana.config`).
"""

__version__ = "0.1.0"
