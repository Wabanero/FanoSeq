"""FanoSeq: sequence trajectories in Fano-structured octonion space."""

from fanoseq.axis_schemes import AXIS_SCHEME_REGISTRY
from fanoseq.encodings import ENCODING_REGISTRY
from fanoseq.octonion import FANO_LINES, Octonion

__all__ = ["AXIS_SCHEME_REGISTRY", "ENCODING_REGISTRY", "FANO_LINES", "Octonion"]
__version__ = "0.1.0"
