"""FanoSeq: sequence trajectories in Fano-structured octonion space."""

__version__ = "0.2.0"

from fanoseq.axis_schemes import (
    AXIS_SCHEME_REGISTRY,
    list_axis_definitions,
    list_axis_schemes,
    validate_axis_scheme_definitions,
)
from fanoseq.encodings import ENCODING_REGISTRY
from fanoseq.encoding_audit import transform_octonion_rc
from fanoseq.fano_plane import FanoPlane, build_fano_line_features, fano_plane_tables
from fanoseq.octonion import FANO_LINES, Octonion

__all__ = [
    "AXIS_SCHEME_REGISTRY",
    "ENCODING_REGISTRY",
    "FANO_LINES",
    "FanoPlane",
    "Octonion",
    "build_fano_line_features",
    "fano_plane_tables",
    "list_axis_definitions",
    "list_axis_schemes",
    "transform_octonion_rc",
    "validate_axis_scheme_definitions",
]
