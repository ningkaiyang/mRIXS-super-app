"""
Package for RIXS line detection algorithms.
"""

from .base import DetectorConfig, LineDetectionResult, BaseLineDetector

__all__ = [
    "DetectorConfig",
    "LineDetectionResult",
    "BaseLineDetector",
    "V8RightSideScanner",
    "DEFAULT_PRESET_ID",
    "get_preset",
    "list_presets"
]

def __getattr__(name):
    """Dynamically import submodules to avoid circular imports or missing files on first load."""
    if name == "V8RightSideScanner":
        from .v8_scanner import V8RightSideScanner
        return V8RightSideScanner
    if name in ("DEFAULT_PRESET_ID", "get_preset", "list_presets"):
        from .presets import DEFAULT_PRESET_ID, get_preset, list_presets
        # Dynamically returning the imported attribute
        if name == "DEFAULT_PRESET_ID":
            return DEFAULT_PRESET_ID
        elif name == "get_preset":
            return get_preset
        elif name == "list_presets":
            return list_presets
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
