# Cómo contribuir

Las reglas completas están en `CLAUDE.md` / `AGENTS.md` en la raíz del repositorio.
Resumen del flujo:

1. Crea una rama: `git checkout -b feat/segmentation-morfologia`.
2. Implementa la función; quita el `pytestmark = pytest.mark.xfail(...)` de su test
   cuando las pruebas pasen.
3. Verifica localmente:
   ```bash
   ruff check . && ruff format --check . && pytest
   ```
4. Commit en español: `feat(segmentation): agrega limpieza morfológica`.
5. Abre un PR; la CI (ruff + pytest) debe pasar antes de fusionar.

## Agregar un módulo nuevo

Todo módulo nuevo en `src/banana/` necesita:

- `tests/test_<modulo>.py`
- `docs/modulos/<modulo>.md` con `::: banana.<modulo>` y una entrada en `mkdocs.yml`.
