"""FanoSeq: sequence trajectories in Fano-structured octonion space."""

from fanoseq.axis_schemes import (
    AXIS_SCHEME_REGISTRY,
    list_axis_definitions,
    list_axis_schemes,
    validate_axis_scheme_definitions,
)
from fanoseq.encodings import ENCODING_REGISTRY
from fanoseq.octonion import FANO_LINES, Octonion

__all__ = [
    "AXIS_SCHEME_REGISTRY",
    "ENCODING_REGISTRY",
    "FANO_LINES",
    "Octonion",
    "list_axis_definitions",
    "list_axis_schemes",
    "validate_axis_scheme_definitions",
]
__version__ = "0.1.0"
