import hashlib
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for line detection algorithm."""
    ref_frac: float = 0.10
    k_rise: float = 4.0
    k_level: float = 2.0
    sustain: int = 8
    y_step: int = 3
    win: int = 6
    peak_win: int = 14
    scan_margin_px: int = 10
    ransac_thresh: float = 4.0
    ransac_iters: int = 3000
    ransac_seed: int = 0
    svd_refine_iters: int = 6

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.k_rise <= 0:
            raise ValueError("k_rise must be > 0")
        if self.k_level < 0:
            raise ValueError("k_level must be >= 0")
        if self.sustain < 1:
            raise ValueError("sustain must be >= 1")
        if self.y_step < 1:
            raise ValueError("y_step must be >= 1")
        if not (0 < self.ref_frac < 1):
            raise ValueError("ref_frac must be in (0, 1)")
        if self.win < 1:
            raise ValueError("win must be >= 1")
        if self.peak_win < 1:
            raise ValueError("peak_win must be >= 1")
        if self.scan_margin_px < 0:
            raise ValueError("scan_margin_px must be >= 0")
        if self.ransac_thresh <= 0:
            raise ValueError("ransac_thresh must be > 0")
        if self.ransac_iters < 1:
            raise ValueError("ransac_iters must be >= 1")
        if self.ransac_seed < 0:
            raise ValueError("ransac_seed must be >= 0")
        if self.svd_refine_iters < 0:
            raise ValueError("svd_refine_iters must be >= 0")

    def fingerprint(self) -> str:
        """Return a unique fingerprint for this configuration."""
        repr_str = (
            f"ref_frac={self.ref_frac:.6f},"
            f"k_rise={self.k_rise:.6f},"
            f"k_level={self.k_level:.6f},"
            f"sustain={self.sustain},"
            f"y_step={self.y_step},"
            f"win={self.win},"
            f"peak_win={self.peak_win},"
            f"scan_margin_px={self.scan_margin_px},"
            f"ransac_thresh={self.ransac_thresh:.6f},"
            f"ransac_iters={self.ransac_iters},"
            f"ransac_seed={self.ransac_seed},"
            f"svd_refine_iters={self.svd_refine_iters}"
        )
        return hashlib.sha256(repr_str.encode('utf-8')).hexdigest()

@dataclass
class LineDetectionResult:
    """Result of a line detection run."""
    fit_ok: bool
    n_candidates: int
    candidates_xy: np.ndarray
    config: DetectorConfig
    failure_reason: str | None = None
    centroid_xy: tuple[float, float] | None = None
    direction_vec: tuple[float, float] | None = None
    angle_deg: float | None = None
    segment_endpoints: tuple[tuple[float, float], tuple[float, float]] | None = None
    detected_support_y_range: tuple[float, float] | None = None
    inliers_xy: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    inlier_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    n_inliers: int = 0
    inlier_fraction: float = 0.0
    residual_median: float | None = None
    residual_p95: float | None = None
    residual_max: float | None = None
    algorithm_id: str = "v8_right_side_scanner"
    fitter_id: str = "ransac_v7"

    def __post_init__(self):
        """Validate the result parameters."""
        if self.candidates_xy.ndim != 2 or self.candidates_xy.shape[1] != 2:
            raise ValueError("candidates_xy must be (N, 2)")
        if self.n_candidates != self.candidates_xy.shape[0]:
            raise ValueError("n_candidates must match candidates_xy.shape[0]")
        if self.fit_ok:
            if self.centroid_xy is None or not (np.isfinite(self.centroid_xy[0]) and np.isfinite(self.centroid_xy[1])):
                raise ValueError("If fit_ok is True, centroid_xy must be valid and finite")
            if self.angle_deg is None or not np.isfinite(self.angle_deg):
                raise ValueError("If fit_ok is True, angle_deg must be valid and finite")
        if self.inlier_mask.size > 0:
            if len(self.inlier_mask) != self.n_candidates:
                raise ValueError("Length of inlier_mask must match n_candidates")
            if self.n_inliers != np.sum(self.inlier_mask):
                raise ValueError("n_inliers must equal the sum of inlier_mask")
        if self.config is None:
            raise ValueError("config cannot be None")

class BaseLineDetector(ABC):
    """Base class for line detection algorithms."""

    @abstractmethod
    def detect(self, prepared: 'PreparedFrame', config: DetectorConfig) -> LineDetectionResult:
        """
        Detect a line in a prepared frame.

        Args:
            prepared: The PreparedFrame object containing image data.
            config: The DetectorConfig parameters.

        Returns:
            LineDetectionResult containing the detection outcome.
        """
        pass
