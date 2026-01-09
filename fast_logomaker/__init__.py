"""
Fast Logomaker - Optimized batch logo generation for sequence logos.
"""

from .core import BatchLogo
from .colors import (
    get_rgb,
    get_color_dict,
    list_color_schemes,
    COLOR_SCHEME_DICT,
)

# Primary class name
FastLogo = BatchLogo

__version__ = "0.1.0"
__all__ = [
    "FastLogo",
    "BatchLogo",  # backwards compatibility alias
    "get_rgb",
    "get_color_dict",
    "list_color_schemes",
    "COLOR_SCHEME_DICT",
]

