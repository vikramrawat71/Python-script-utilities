"""
Image Extractor — Crops and saves image regions detected during layout analysis.
"""

import logging
import os
import cv2
import numpy as np
from PIL import Image as PILImage
from .models import Region

logger = logging.getLogger(__name__)


def extract_images(frame: np.ndarray, image_regions: list[Region],
                   output_dir: str | None = None) -> list[PILImage.Image]:
    images = []
    for i, region in enumerate(image_regions):
        try:
            crop = region.crop_from(frame)
            if crop.size == 0:
                logger.warning(f"  Image region {i} has zero area, skipping")
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            pil_img = _enhance_image(pil_img)
            images.append(pil_img)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"image_{i:03d}.png")
                pil_img.save(save_path)
                logger.debug(f"  Saved image {i} to {save_path}")
        except Exception as e:
            logger.warning(f"  Failed to extract image region {i}: {e}")
    logger.debug(f"  Extracted {len(images)} images from frame")
    return images


def _enhance_image(img: PILImage.Image) -> PILImage.Image:
    from PIL import ImageEnhance, ImageStat
    stat = ImageStat.Stat(img)
    if all(s < 30 for s in stat.stddev[:3]):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)
    return img


def save_images_to_dir(images: list[PILImage.Image], output_dir: str,
                       prefix: str = "page") -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"{prefix}_{i:03d}.png")
        img.save(path, "PNG")
        paths.append(os.path.abspath(path))
        logger.debug(f"  Saved: {path}")
    return paths
