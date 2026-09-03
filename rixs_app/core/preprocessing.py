import numpy as np
import scipy.ndimage
import cv2
from dataclasses import dataclass

def denoise_image(
    img: np.ndarray,
    clip: bool = True,
    despike: bool = True,
    anscombe: bool = True,
    bilateral: bool = True,
    inverse_anscombe: bool = True,
    mad_threshold: float = 5.0,
    d: int = 5,
    sigma_color: float = 1.5,
    sigma_space: float = 3.0
) -> np.ndarray:
    """
    Denoise a 2D spectroscopic frame image using a robust, multi-step pipeline.

    Physics Context:
    Raw RIXS CCD scans suffer from severe Poisson shot noise, read noise, and cosmic ray strikes.
    Standard high-frequency metrics fail because they mistakenly latch onto residual
    noise spikes instead of the actual elastic line. This denoising pipeline aims to sanitize
    the raw data before evaluation or mathematical isolation.

    Processing Steps:
    1. Clipping: Eliminates negative baseline artifacts.
    2. MAD Despiking: Removes cosmic ray strikes using Median Absolute Deviation thresholding.
    3. Anscombe VST: Variance-Stabilizing Transformation converts Poisson noise to approximately
       constant-variance Gaussian noise.
    4. Bilateral Filtering: Smooths the VST-transformed image while preserving structural edges
       (the spectroscopic line).
    5. Inverse Anscombe VST: Applies an algebraic inverse to return to the original intensity scale.

    Args:
        img: 2D numpy array containing the raw detector image.
        clip: Boolean indicating whether to clip negative values to zero.
        despike: Boolean indicating whether to apply MAD despiking for cosmic rays.
        anscombe: Boolean indicating whether to apply the Anscombe variance-stabilizing transformation.
        bilateral: Boolean indicating whether to apply the bilateral filter.
        inverse_anscombe: Boolean indicating whether to apply the inverse Anscombe transformation.
        mad_threshold: Float multiplier for the MAD threshold during despiking.
        d: Integer diameter of each pixel neighborhood for the bilateral filter.
        sigma_color: Float filter sigma in the color space for bilateral filtering.
        sigma_space: Float filter sigma in the coordinate space for bilateral filtering.

    Returns:
        np.ndarray: The denoised 2D image as a float32 array.

    Raises:
        ValueError: If the input is not a 2D numpy array or is empty.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if img.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("Input array cannot be empty")

    # Cast first to ensure any overflow becomes inf and is sanitized
    img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # a. Clipping
    if clip:
        img = np.clip(img, 0.0, None)

    # b. MAD despiking
    if despike:
        median_img = scipy.ndimage.median_filter(img, size=3)
        dev = img - median_img
        mad = np.median(np.abs(dev))
        if mad < 1e-6:
            mad = np.std(dev, dtype=np.float64)
            if np.isnan(mad) or np.isinf(mad) or mad <= 0.0:
                mad = 1e-6
        if mad > 1e-6:
            threshold = mad_threshold * 1.4826 * mad
            img = np.where(np.abs(dev) > threshold, median_img, img)

    # c. Anscombe VST
    if anscombe:
        img = 2.0 * np.sqrt(np.maximum(img + 0.375, 0.0))

    # d. Bilateral Filter
    if bilateral:
        img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    # e. Inverse Anscombe VST
    if inverse_anscombe:
        img = (img / 2.0)**2 - 0.375

    # Final clipping pass when clip=True to prevent negative inverse Anscombe output
    if clip:
        img = np.clip(img, 0.0, None)

    return img.astype(np.float32)


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
