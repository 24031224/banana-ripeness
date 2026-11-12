# `banana.classifier`

## Propósito

Convierte un `FeatureSet` en un `Verdict` explicable. Hay dos clasificadores que
devuelven el mismo tipo de salida.

## Entradas y salidas

| Función | Entrada | Salida |
|---|---|---|
| `classify_rules` | `FeatureSet` + `ClassifierConfig` | `Verdict` (clase, confianza en [0, 1], razones) |
| `train_tree` | ruta a un CSV etiquetado | `DecisionTreeClassifier` entrenado |
| `classify_tree` | árbol + `FeatureSet` | `Verdict` |
| `export_tree_rules` | árbol | texto con las reglas |

Errores: `ClassifierError` (hereda de `BananaError`) si el CSV es inválido.

## Reglas (método principal)

`classify_rules(features, cfg)` evalúa en orden; la primera clase que aplica gana:

| Prioridad | Clase | Condición (umbrales en `classifier:` de `config/thresholds.yaml`) |
|---|---|---|
| 1 | No comestible | `brown_ratio >= no_comestible_brown_ratio_min` **o** `dark_spot_ratio >= no_comestible_dark_ratio_min` |
| 2 | Inmaduro | `green_ratio >= inmaduro_green_ratio_min` |
| 3 | Comestible | ninguna de las anteriores |

### Cómo se explica la decisión

`Verdict.reasons` lista las reglas con valor y umbral:

- No comestible / Inmaduro: las condiciones que **se cumplieron**,
  p. ej. `("brown_ratio=0.52 >= 0.40",)`.
- Comestible: como se asigna por descarte, se listan las tres condiciones que
  **no** se cumplieron, p. ej. `("brown_ratio=0.10 < 0.40", ...)`.

### Confianza

`margin_confidence(valor, umbral)` = `|valor - umbral| / max(umbral, 1 - umbral)`,
en [0, 1]: 0 exactamente en el umbral y 1 en el extremo opuesto. Para
"No comestible" se toma la regla disparada de mayor margen; para "Comestible",
el menor margen de las tres reglas descartadas (qué tan cerca estuvo de cambiar).

```python
from banana.classifier import classify_rules

verdict = classify_rules(features, thresholds.classifier)
print(verdict.label, verdict.confidence, verdict.reasons)
```

## Árbol de decisión (comparación y exploración)

Un `DecisionTreeClassifier(max_depth=3)` de scikit-learn entrenado con un CSV.

CSV esperado: encabezado con las columnas `hue_mean, hue_std, saturation_mean,
value_mean, green_ratio, yellow_ratio, brown_ratio, dark_spot_ratio, label`.
`label` acepta `Inmaduro`/`Comestible`/`No comestible` o el nombre de carpeta del
dataset (`inmaduro`/`comestible`/`no_comestible`). Recuerda que `hue_*` está en
escala OpenCV (H en 0-179).

```python
from banana.classifier import train_tree, classify_tree, export_tree_rules

tree = train_tree("data/features.csv")
print(export_tree_rules(tree))  # reglas del árbol como texto
verdict = classify_tree(tree, features)  # reasons = camino raíz → hoja
```

## Pipeline de un solo paso

`banana.pipeline.analyze(path_or_array)` devuelve
`(Verdict, FeatureSet, imagen con máscara superpuesta)`; ver
[`pipeline`](pipeline.md).

## Decisiones de diseño

- **Reglas como método principal:** son explicables y se calibran a mano desde
  `config/thresholds.yaml`; el proyecto no usa deep learning como método principal.
- **Prioridad No comestible > Inmaduro > Comestible:** un plátano con manchas
  oscuras es un riesgo aunque tenga algo de verde, así que se descarta primero.
- **Confianza por margen al umbral:** es simple y refleja qué tan cerca estuvo la
  decisión de cambiar de clase, sin necesidad de probabilidades entrenadas.
- **Árbol con `max_depth=3` y semilla fija:** se mantiene legible
  (`export_tree_rules`) y reproducible; solo sirve para comparar e inspirar umbrales.
- **`sklearn` se importa dentro de las funciones:** las reglas funcionan sin cargar
  scikit-learn, que solo se usa con el árbol.
- **Umbrales nunca en código:** vienen de `ClassifierConfig`.

## Referencia

::: banana.classifier
