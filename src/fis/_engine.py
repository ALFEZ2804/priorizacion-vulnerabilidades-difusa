"""Motor de inferencia difusa para el FIS jerárquico.

Capa de abstracción única sobre `skfuzzy.control`. El resto del repo
(baselines, evaluación, web app) NO debe importar `skfuzzy.control`
directamente: solo ve `FuzzyInference`. Esto mantiene vivos tres
caminos de extensión sin tocar código fuera de `src/fis/`:

- Posible extensión: sustituir Mamdani tipo-1 por Interval Type-2.
- Posible extensión: añadir un FIS Sugeno alternativo y compararlos empíricamente.
- Posible extensión: módulo XAI sobre los grados de activación de las reglas.

Razón estructural: si la inferencia se escribiese con
`ctrl.ControlSystemSimulation` esparcida por el pipeline, las tres extensiones futuras quedarían cerradas por arquitectura (no por decisión
consciente).

Dos vías de inferencia, numéricamente equivalentes:

- `infer(**inputs)`: un caso vía `skfuzzy.control`. Implementación de
  referencia, legible y directa; se usa en tests y casos sueltos.
- `infer_batch(**arrays)`: N casos a la vez con una reimplementación
  vectorizada de Mamdani en numpy. `skfuzzy.control`
  instancia un simulador por caso y tarda ~21 min sobre los 92k CVEs del
  unified; la versión vectorizada calcula los mismos valores en ~15 s.
  `tests/test_fis_engine.py::test_infer_batch_matches_infer` verifica que
  ambas coinciden punto a punto.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from skfuzzy import control as ctrl

# Tamaño de bloque de filas en infer_batch: acota la matriz intermedia
# (bloque × universo_consecuente) en memoria. 5000 × 1001 ≈ 40 MB por bloque.
_BATCH_CHUNK = 5000

# Epsilon de la división final del centroide (mismo que usa skfuzzy).
_EPS = np.finfo(float).eps


def _centroid_batch(universe: np.ndarray, mfx: np.ndarray) -> np.ndarray:
    """Centroide por integración trapezoidal, vectorizado sobre las filas de `mfx`.

    Replica exactamente `skfuzzy.defuzzify.defuzz.centroid`: área y momento
    exactos de cada trapecio/triángulo entre muestras consecutivas del universo,
    pero para las N filas de `mfx` a la vez en lugar de un bucle por muestra.
    `universe` es estrictamente creciente, así que el caso `x1 == x2` de skfuzzy
    no se da. Una fila con algún NaN propaga NaN; una fila de área nula da 0.
    """
    x1 = universe[:-1]
    dx = universe[1:] - x1
    y1 = mfx[:, :-1]
    y2 = mfx[:, 1:]

    # La fórmula del trapecio cubre rectángulo y triángulos como casos
    # particulares (algebraicamente idéntica a las cuatro ramas de skfuzzy).
    # Donde y1+y2 == 0 el área es 0 y el momento no contribuye, así que solo hay
    # que evitar el 0/0 de la división.
    paired = y1 + y2
    safe = np.where(paired == 0.0, 1.0, paired)
    area = 0.5 * dx * paired
    moment = (2.0 / 3.0 * dx * (y2 + 0.5 * y1)) / safe + x1
    return (moment * area).sum(axis=1) / np.fmax(area.sum(axis=1), _EPS)


class FuzzyInference:
    """Sub-FIS Mamdani encapsulado.

    Parámetros
    ----------
    antecedents
        Variables de entrada (`ctrl.Antecedent`) con sus funciones de
        membresía ya definidas, sea con `automf` o asignación manual.
    consequent
        Variable de salida (`ctrl.Consequent`) con sus MFs ya definidas.
    rules
        Reglas Mamdani (`ctrl.Rule`) que conectan antecedentes y
        consecuente.

    Ejemplo
    -------
    >>> import numpy as np
    >>> from skfuzzy import control as ctrl
    >>> x = ctrl.Antecedent(np.arange(0, 11, 1), "x")
    >>> y = ctrl.Consequent(np.arange(0, 11, 1), "y")
    >>> x.automf(3); y.automf(3)
    >>> rule = ctrl.Rule(x["poor"], y["poor"])
    >>> fis = FuzzyInference(antecedents=[x], consequent=y, rules=[rule])
    >>> fis.infer(x=0.0)  # float coherente
    """

    def __init__(
        self,
        antecedents: Iterable[ctrl.Antecedent],
        consequent: ctrl.Consequent,
        rules: Iterable[ctrl.Rule],
    ) -> None:
        self._antecedents = list(antecedents)
        self._consequent = consequent
        self._rules = list(rules)

        # ControlSystem: grafo simbólico de reglas (construcción barata).
        # ControlSystemSimulation: instancia ejecutable: la guardamos
        # para reutilizarla en cada llamada a infer(). Sin esto, cada
        # CVE pagaría el coste de reconstruir el grafo de inferencia.
        self._system = ctrl.ControlSystem(self._rules)
        self._simulation = ctrl.ControlSystemSimulation(self._system)

    @property
    def antecedent_names(self) -> list[str]:
        return [a.label for a in self._antecedents]

    @property
    def consequent_name(self) -> str:
        return self._consequent.label

    def infer(self, **inputs: float) -> float:
        """Ejecuta una inferencia crisp y devuelve el valor defuzzificado.

        Los nombres de los parámetros deben coincidir con los `label`
        de los antecedentes:

            fis.infer(exploitability_subscore=2.4, impact_subscore=5.0)

        Si los inputs caen en una zona del espacio que ninguna regla
        cubre, `skfuzzy` puede devolver NaN. Lo dejamos pasar tal cual:
        silenciarlo a un valor por defecto falsearía el ranking. Si
        ocurre con datos reales, hay que revisar la cobertura de las MFs.
        """
        for name, value in inputs.items():
            self._simulation.input[name] = float(value)
        self._simulation.compute()
        return float(self._simulation.output[self._consequent.label])

    def infer_batch(self, **arrays: np.ndarray) -> np.ndarray:
        """Versión vectorizada de `infer` para puntuar N casos a la vez.

        Reproduce exactamente la inferencia Mamdani de `infer()` con álgebra
        de numpy en vez del bucle de `skfuzzy.control` (que instancia un
        simulador por caso). Los cinco pasos de Mamdani, vectorizados sobre
        las N filas a la vez:

          1. Fuzzificar: grado de pertenencia de cada input a cada término
             antecedente, vía `np.interp` sobre la MF del término.
          2. Disparo de regla: AND = mínimo de los grados de sus antecedentes.
          3. Implicación: recortar la MF del consecuente al nivel de disparo
             (`min`), escalada por el peso de la regla.
          4. Agregación: combinar las reglas por `max` sobre el universo del
             consecuente.
          5. Defuzzificar: centroide por integración trapezoidal sobre el
             universo del consecuente (idéntico a `skfuzzy`, ver `_centroid_batch`).

        Cada array debe tener longitud N y su nombre coincidir con el `label`
        del antecedente. Los NaN se propagan: un caso con algún input NaN
        devuelve NaN (igual que el guard explícito que tenían los `compute_*`).
        """
        names = list(arrays)
        n = len(arrays[names[0]])
        inputs = {name: np.asarray(values, dtype=float) for name, values in arrays.items()}
        cons_universe = self._consequent.universe
        out = np.empty(n, dtype=float)

        for start in range(0, n, _BATCH_CHUNK):
            stop = min(start + _BATCH_CHUNK, n)
            rows = stop - start

            # Pasos 1-2: disparo de cada regla (AND = mínimo de sus antecedentes),
            # acumulando el máximo por término del consecuente. Agregar las reglas
            # por término aquí (en vez de recortar la MF una vez por regla) es
            # equivalente (max(min(fᵢ, mf)) = min(max fᵢ, mf)) pero recorta la MF
            # una sola vez por término: 3-4 veces por bloque en lugar de 9 o 27.
            term_activation: dict[str, tuple[np.ndarray, object]] = {}
            for rule in self._rules:
                firing = np.ones(rows)
                for term in rule.antecedent_terms:  # AND = mínimo
                    var = term.parent
                    mu = np.interp(inputs[var.label][start:stop], var.universe, term.mf)
                    firing = np.minimum(firing, mu)
                weighted = rule.consequent[0]  # nuestras reglas: un solo consecuente
                activation = firing * weighted.weight
                label = weighted.term.label
                if label in term_activation:
                    np.maximum(term_activation[label][0], activation, out=term_activation[label][0])
                else:
                    term_activation[label] = (activation, weighted.term)

            # Pasos 3-4: implicación (recorte por min) y agregación (max).
            aggregated = np.zeros((rows, cons_universe.size))
            for activation, term in term_activation.values():
                clipped = np.minimum(activation[:, None], term.mf[None, :])
                np.maximum(aggregated, clipped, out=aggregated)

            # Paso 5: defuzzificar por centroide.
            out[start:stop] = _centroid_batch(cons_universe, aggregated)
        return out
