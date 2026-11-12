# `banana.segmentation`

Separa el plátano del fondo con un rango HSV, limpieza morfológica y el mayor contorno.

## Flujo

1. `color_mask`: `cv2.inRange` con `segmentation.banana_hsv` (H en 0-179).
2. `clean_mask`: apertura y luego cierre con kernel elíptico (`morph_kernel`, `morph_iterations`).
3. `largest_component`: conserva el contorno externo de mayor área, relleno.
4. `segment`: calcula caja, área y proporción; lanza `SegmentationError` si la máscara está
   vacía o el área es menor que `min_area_ratio`.

!!! warning "Rango amplio"
    `banana_hsv` debe cubrir cáscara verde, amarilla **y** café. Si un plátano muy maduro
    queda fuera de la máscara, el clasificador nunca lo verá café.

::: banana.segmentation
