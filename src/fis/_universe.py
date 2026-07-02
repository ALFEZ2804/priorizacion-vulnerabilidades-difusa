"""Helper de construcción de universos para antecedentes/consecuentes del FIS.

Compartido por todos los sub-FIS (`severity`, `threat`, `priority`) para no
duplicar la rejilla del dominio. Vive aquí (y no en cada módulo) porque la
sutileza de usar `linspace` en vez de `arange` es la misma para los tres.
"""

from __future__ import annotations

import numpy as np


def universe(lo: float, hi: float, step: float) -> np.ndarray:
    """Rejilla equiespaciada con extremos exactos (sin drift de floating point).

    `np.arange` con stops decimales puede dejar puntos ligeramente fuera del
    rango y producir μ=0 en el borde, lo que rompe la cobertura del soporte de
    las MFs con vértice en el extremo (p. ej. `high=[2.5, 3.9, 3.9]` en FIS-1 o
    `high=[0.7, 0.9, 1, 1]` en FIS-2). `linspace` garantiza que el último punto
    es exactamente `hi`.
    """
    n = int(round((hi - lo) / step)) + 1
    return np.linspace(lo, hi, n)
