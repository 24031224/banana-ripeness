# `banana.acquisition`

Lectura de imágenes desde archivo, bytes, cámara y datasets etiquetados. Todas las
funciones devuelven imágenes **BGR `uint8`** y lanzan `ImageLoadError` si algo falla.

## Rutas de Windows

`load_image` no usa `cv2.imread` (falla con tildes, ñ o espacios en rutas de Windows).
En su lugar lee los bytes con `np.fromfile` y decodifica con `cv2.imdecode`.

## Dataset

`iter_dataset` espera `<raíz>/{inmaduro,comestible,no_comestible}/*.jpg`, ignora
subcarpetas desconocidas y archivos con otras extensiones, y devuelve las muestras
ordenadas por ruta.

::: banana.acquisition
