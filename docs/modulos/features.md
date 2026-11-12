# `banana.features`

Características de color calculadas **solo dentro de la máscara** del plátano.

## Características

| Campo | Cálculo |
|---|---|
| `hue_mean`, `hue_std` | Media y desviación de H (escala 0-179, media lineal) |
| `saturation_mean`, `value_mean` | Medias de S y V (0-255) |
| `green_ratio`, `yellow_ratio`, `brown_ratio` | Fracción de píxeles del plátano dentro de cada rango HSV |
| `dark_spot_ratio` | Fracción con `V < dark_spot_value_max` |
| `hue_histogram` | Histograma de H normalizado, `hue_bins` cubetas sobre 0-179 |

Las proporciones están en [0, 1] respecto a los píxeles del plátano, no de la imagen.
Los rangos de verde, amarillo y café **pueden solaparse** (p. ej. café y amarillo comparten
H 20), por lo que las proporciones no tienen por qué sumar 1.

::: banana.features
