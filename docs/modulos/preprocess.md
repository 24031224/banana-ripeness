# `banana.preprocess`

Redimensionado, suavizado, balance de blancos y conversión a HSV (H 0-179).

## Flujo

1. `resize_to_width`: ancho fijo (`target_width`), proporción conservada, `INTER_AREA`.
2. `denoise`: suavizado gaussiano (`blur_kernel`; 1 = sin suavizado).
3. `apply_white_balance`: `gray_world` escala cada canal hacia la media global; `none` no hace nada.
4. `to_hsv`: conversión de OpenCV (**H en 0-179**, S y V en 0-255).

Todos los parámetros vienen de la sección `preprocess` de `config/thresholds.yaml`.

!!! note "Orden importante"
    El balance de blancos se aplica **después** del suavizado y **antes** de pasar a HSV,
    así los rangos de color de las etapas siguientes ya ven la imagen corregida.

::: banana.preprocess
