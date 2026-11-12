# Configuración de umbrales

Todos los umbrales viven en `config/thresholds.yaml` y se cargan con
`banana.config.load_thresholds()`.

!!! warning "OpenCV usa H en 0-179"
    Si tomas un tono de una herramienta que usa grados (0-360), divídelo entre 2.
    Un amarillo de 54° es `H = 27` en OpenCV.

| Sección | Clave | Significado |
|---|---|---|
| `preprocess` | `target_width` | Ancho al que se redimensiona la imagen |
| | `blur_kernel` | Kernel gaussiano (impar) |
| | `white_balance` | `none` o `gray_world` |
| `segmentation` | `banana_hsv` | Rango que cubre la cáscara verde, amarilla y café |
| | `morph_kernel`, `morph_iterations` | Limpieza morfológica |
| | `min_area_ratio` | Área mínima del plátano (fracción de la imagen) |
| `features` | `hue_bins` | Cubetas del histograma de H |
| | `green`, `yellow`, `brown` | Rangos HSV por color |
| | `dark_spot_value_max` | V por debajo del cual un píxel es mancha |
| `classifier` | `no_comestible_brown_ratio_min` | Fracción café mínima para "No comestible" |
| | `no_comestible_dark_ratio_min` | Fracción de manchas mínima para "No comestible" |
| | `inmaduro_green_ratio_min` | Fracción verde mínima para "Inmaduro" |

## Recalibrar

```bash
python scripts/calibrate.py --dataset data/raw
python scripts/evaluate.py  --dataset data/test
```

Cada cambio de umbrales va en su propio commit, con el resultado de la evaluación en
la descripción.
