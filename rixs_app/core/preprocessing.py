import numpy as np
import scipy.ndimage
import cv2
from dataclasses import dataclass

from rixs_app.core.zeroth_order import denoise_image

@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Configuration for frame preprocessing before line detection.
    """
    edge_crop_px: int = 100
    smooth_sigma: float = 2.5
    mad_threshold: float = 5.0
    bilateral_d: int = 5
    bilateral_sigma_color: float = 1.5
    bilateral_sigma_space: float = 3.0

@dataclass(frozen=True)
class CropTransform:
    """
    Stores the crop coordinates and provides methods to transform
    between original and cropped spaces.
    """
    crop_top: int
    crop_left: int
    crop_bottom: int
    crop_right: int
    original_shape: tuple[int, int]
    cropped_shape: tuple[int, int]

    def cropped_to_original(self, x: float, y: float) -> tuple[float, float]:
        """Convert a single coordinate pair from cropped space to original space."""
        return x + self.crop_left, y + self.crop_top

    def original_to_cropped(self, x: float, y: float) -> tuple[float, float]:
        """Convert a single coordinate pair from original space to cropped space."""
        return x - self.crop_left, y - self.crop_top

    def cropped_to_original_array(self, xy: np.ndarray) -> np.ndarray:
        """Convert an (N, 2) array of coordinates from cropped space to original space."""
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("Input array must have shape (N, 2)")
        result = xy.copy()
        result[:, 0] += self.crop_left
        result[:, 1] += self.crop_top
        return result

@dataclass
class PreparedFrame:
    """
    Contains all intermediate representations of a frame during preprocessing.
    """
    raw: np.ndarray
    cropped_raw: np.ndarray
    denoised: np.ndarray
    row_smoothed: np.ndarray
    gradient: np.ndarray
    crop_transform: CropTransform
    config: PreprocessingConfig

def compute_gradients(D: np.ndarray) -> np.ndarray:
    """
    Compute image gradients matching the V8 algorithm's gradients() function.

    Args:
        D: Denoised image array.

    Returns:
        Gradient magnitude array.
    """
    g = cv2.GaussianBlur(D, (5, 5), 1.5)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    return np.sqrt(gx * gx + gy * gy)

def prepare_frame(img: np.ndarray, config: PreprocessingConfig = None) -> PreparedFrame:
    """
    Preprocess a raw image frame for line detection.

    IMPORTANT: The cropping must happen before denoising to match the original V8 algorithm.

    Args:
        img: Raw 2D input image.
        config: Preprocessing configuration parameters.

    Returns:
        PreparedFrame object containing original, cropped, denoised, smoothed, and gradient images.
    """
    if config is None:
        config = PreprocessingConfig()

    img = img.astype(np.float32)
    H, W = img.shape

    crop = config.edge_crop_px
    if H > 2 * crop + 50 and W > 2 * crop + 50:
        crop_top, crop_bottom = crop, crop
        crop_left, crop_right = crop, crop
        cropped_raw = img[crop:H - crop, crop:W - crop]
    else:
        crop_top, crop_bottom = 0, 0
        crop_left, crop_right = 0, 0
        cropped_raw = img

    cropped_raw = np.ascontiguousarray(cropped_raw)

    crop_transform = CropTransform(
        crop_top=crop_top,
        crop_left=crop_left,
        crop_bottom=crop_bottom,
        crop_right=crop_right,
        original_shape=(H, W),
        cropped_shape=cropped_raw.shape
    )

    # Denoise using the common core implementation
    denoised = denoise_image(
        cropped_raw,
        mad_threshold=config.mad_threshold,
        d=config.bilateral_d,
        sigma_color=config.bilateral_sigma_color,
        sigma_space=config.bilateral_sigma_space
    )

    # Row-wise Gaussian smoothing then rolling-min background subtraction
    # The rolling-min removes noise haze without affecting the denoised image used for Gaussian evaluation
    row_smoothed_raw = scipy.ndimage.gaussian_filter1d(denoised, sigma=config.smooth_sigma, axis=1)
    row_smoothed_bg = scipy.ndimage.minimum_filter1d(row_smoothed_raw, size=150, axis=1)
    row_smoothed = np.maximum(0.0, row_smoothed_raw - row_smoothed_bg).astype(np.float32)

    # Compute gradients
    gradient = compute_gradients(denoised)

    return PreparedFrame(
        raw=img,
        cropped_raw=cropped_raw,
        denoised=denoised,
        row_smoothed=row_smoothed,
        gradient=gradient,
        crop_transform=crop_transform,
        config=config
    )
