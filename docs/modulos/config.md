# `banana.config`

Carga y validación de `config/thresholds.yaml` en dataclasses tipadas e inmutables.

## Uso

```python
from banana.config import load_thresholds

cfg = load_thresholds()  # ruta por defecto: <raíz del repo>/config/thresholds.yaml
cfg = load_thresholds("otra/ruta.yaml")  # o una ruta explícita
```

La ruta por defecto (`DEFAULT_THRESHOLDS_PATH`) se calcula a partir de la ubicación del
paquete, no del directorio de trabajo, así que funciona desde cualquier carpeta.

## Validaciones

Cualquier problema lanza `ConfigError` con un mensaje que nombra la clave (p. ej.
`features.dark_spot_value_max`):

- el archivo no existe, no se puede leer o no es YAML válido;
- falta una sección o una clave;
- **H fuera de 0-179** (OpenCV usa grados / 2, no 0-360) o S/V fuera de 0-255;
- `lower > upper` en algún componente de un rango HSV;
- kernels pares (`blur_kernel`, `morph_kernel`) o enteros que no lo son;
- proporciones fuera de [0, 1] o `white_balance` desconocido.

::: banana.config
