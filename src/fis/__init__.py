"""Sub-paquete del FIS jerárquico: único punto de contacto con skfuzzy."""

from src.fis._engine import FuzzyInference
from src.fis.hierarchical import compute_priority

__all__ = ["FuzzyInference", "compute_priority"]
