# `banana.pipeline`

## Propósito

Orquestador de extremo a extremo; único punto de entrada para la app de Streamlit
y los scripts. Encadena `preprocess → segment → extract_features → classify_rules`.

## Entradas y salidas

| Función / método | Entrada | Salida |
|---|---|---|
| `BananaPipeline.from_yaml(path)` | ruta al YAML de umbrales (por defecto `config/thresholds.yaml`) | `BananaPipeline` |
| `BananaPipeline.run(image)` | imagen BGR `uint8` | `PipelineResult` con todas las etapas intermedias |
| `BananaPipeline.classify_path(path)` | ruta a una imagen | `Verdict` |
| `overlay_mask(image, mask)` | imagen BGR + máscara 0/255 del mismo tamaño | copia con el plátano resaltado |
| `analyze(path_or_array, pipeline=None)` | ruta o imagen BGR | `(Verdict, FeatureSet, overlay)` |

Errores: `ImageLoadError` si la entrada no es una imagen válida y
`SegmentationError` si no se detecta un plátano.

## Ejemplo de uso

```python
from banana.pipeline import analyze

verdict, features, overlay = analyze("data/samples/comestible_01.jpg")
print(verdict.label, verdict.confidence, verdict.reasons)
```

## Decisiones de diseño

- **`analyze` para la app:** una sola llamada devuelve lo que la interfaz necesita
  (veredicto, características e imagen), así Streamlit no conoce las etapas.
- **Pipeline por defecto en caché (`lru_cache`):** el YAML se lee una vez por
  proceso, no en cada imagen.
- **Overlay sobre la imagen preprocesada:** la máscara está en las coordenadas de
  esa imagen (puede estar redimensionada), así que se dibuja sobre ella y no sobre la original.
- **`overlay_mask` no modifica la entrada:** trabaja sobre una copia y valida tamaños.
- **Color y opacidad como constantes de módulo:** son de presentación, no umbrales de clasificación.

## Referencia

::: banana.pipeline
