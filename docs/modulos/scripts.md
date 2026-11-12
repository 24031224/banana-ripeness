# Scripts de dataset y árbol de decisión

## Propósito

Dos scripts de `scripts/` que generan la comparación **reglas vs árbol de decisión**:

1. `extract_dataset.py` convierte las imágenes de `data/<split>/` en un CSV de características.
2. `train_tree.py` entrena un árbol (`max_depth=3`, `random_state=42`) con el CSV de train,
   lo evalúa en test y lo compara con `classify_rules` sobre el mismo test.

## Estructura de datos esperada

```text
data/
  train/
    inmaduro/*.jpg|png
    comestible/*.jpg|png
    no_comestible/*.jpg|png
  test/
    (misma estructura)
```

Si falta la carpeta o no hay imágenes, los scripts terminan con código 1 y un mensaje que
muestra esta estructura.

## Uso

```bash
python scripts/extract_dataset.py --split train   # → data/features_train.csv
python scripts/extract_dataset.py --split test    # → data/features_test.csv
python scripts/train_tree.py                      # → docs/results/
```

Las imágenes cuya segmentación falla se registran en `data/extract_<split>.log` y se omiten.

## Columnas del CSV

| Columna | Significado |
|---|---|
| `ruta`, `etiqueta` | Imagen y carpeta de clase (`inmaduro`, `comestible`, `no_comestible`) |
| `pct_verde`, `pct_amarillo`, `pct_marron` | Proporción de píxeles del plátano en cada rango HSV de `features` |
| `pct_oscuro` | Proporción con V menor que `features.dark_spot_value_max` |
| `n_manchas` | Número de manchas oscuras (componentes conexas tras una apertura morfológica) |
| `area_manchas_pct` | Área de esas manchas respecto al plátano |
| `other_ratio` | Proporción fuera de verde, amarillo y marrón |

Todas las `pct_*`, `area_manchas_pct` y `other_ratio` son **fracciones en [0, 1]**, igual que
en `FeatureSet` (no porcentajes 0-100). En el HSV de OpenCV, H va de 0 a 179.

## Resultados en `docs/results/`

`matriz_confusion_arbol.png`, `reporte_arbol.md`, `importancia_caracteristicas.md`,
`arbol.txt`, `arbol.png` y `comparacion.md` (tabla reglas vs árbol con el recall de
`no_comestible` destacado).

## Decisiones de diseño

- **Sin umbrales nuevos:** los rangos de color y el umbral de manchas vienen de
  `config/thresholds.yaml`; el kernel de la apertura de manchas reutiliza
  `segmentation.morph_kernel`. Solo `max_depth` (`banana.classifier.TREE_MAX_DEPTH`) y la
  semilla 42 son constantes de los scripts.
- **El árbol usa las columnas de este CSV**, no `banana.classifier.train_tree`, que espera
  otras columnas (`hue_mean`, `green_ratio`, ...). No se modifica `classifier.py`.
- **Reglas sobre el CSV:** `classify_rules` solo usa verde, marrón y oscuro, que están en el
  CSV; los estadísticos HSV no usados se rellenan con 0.0.
- **Solo `BananaPipeline`:** los scripts obtienen HSV, máscara y características de
  `BananaPipeline.run`.

## Referencia

::: extract_dataset

::: train_tree
