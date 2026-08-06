"""Registry for line-finding presets.

Defines available detector configurations and a default.
"""

from typing import Dict, Tuple, List
from rixs_app.core.line_finding.base import DetectorConfig

PRESETS: Dict[str, Tuple[str, DetectorConfig]] = {
    "v8_g10_r4_l2_s8": (
        "V8 baseline \u2014 g10_r4_l2_s8",
        DetectorConfig(
            ref_frac=0.10,
            k_rise=4.0,
            k_level=2.0,
            sustain=8,
            y_step=3,
            win=6,
            peak_win=14,
            scan_margin_px=10,
            ransac_thresh=4.0,
            ransac_iters=3000,
            ransac_seed=0,
            svd_refine_iters=6,
        ),
    ),
}

DEFAULT_PRESET_ID = "v8_g10_r4_l2_s8"

def get_preset(preset_id: str) -> Tuple[str, DetectorConfig]:
    """Retrieve a preset by its ID.

    Args:
        preset_id (str): The identifier of the preset to retrieve.

    Returns:
        Tuple[str, DetectorConfig]: A tuple containing the preset's display name and its configuration.

    Raises:
        KeyError: If the preset_id does not exist in the registry.
    """
    return PRESETS[preset_id]

def list_presets() -> List[Tuple[str, str]]:
    """List all available presets.

    Returns:
        List[Tuple[str, str]]: A list of tuples, each containing a preset ID and its display name.
    """
    return [(preset_id, display_name) for preset_id, (display_name, _) in PRESETS.items()]
