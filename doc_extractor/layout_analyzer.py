"""
Layout Analyzer — Detects document regions (header, footer, body, images)
within a frame using OpenCV contour detection and positional heuristics.
"""

import logging
import cv2
import numpy as np
from .models import Region, PageLayout

logger = logging.getLogger(__name__)

HEADER_FRACTION = 0.12
FOOTER_FRACTION = 0.12
MIN_IMAGE_AREA_RATIO = 0.005
MAX_IMAGE_AREA_RATIO = 0.60
MIN_IMAGE_ASPECT = 0.2
MAX_IMAGE_ASPECT = 5.0
CONTENT_MARGIN_FRACTION = 0.03


def _detect_content_bounds(frame: np.ndarray) -> Region:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        if (cw * ch) > (w * h * 0.3):
            margin_x = int(w * CONTENT_MARGIN_FRACTION)
            margin_y = int(h * CONTENT_MARGIN_FRACTION)
            return Region(x=max(0, x + margin_x), y=max(0, y + margin_y),
                          width=min(w, cw - 2 * margin_x), height=min(h, ch - 2 * margin_y))

    margin_x = int(w * CONTENT_MARGIN_FRACTION)
    margin_y = int(h * CONTENT_MARGIN_FRACTION)
    return Region(x=margin_x, y=margin_y, width=w - 2 * margin_x, height=h - 2 * margin_y)


def _detect_image_regions(frame: np.ndarray, body_region: Region) -> list[Region]:
    body_crop = body_region.crop_from(frame)
    gray = cv2.cvtColor(body_crop, cv2.COLOR_BGR2GRAY)
    frame_area = frame.shape[0] * frame.shape[1]
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = w / h if h > 0 else 0
        area_ratio = area / frame_area
        if area_ratio < MIN_IMAGE_AREA_RATIO or area_ratio > MAX_IMAGE_AREA_RATIO:
            continue
        if aspect < MIN_IMAGE_ASPECT or aspect > MAX_IMAGE_ASPECT:
            continue
        region_crop = body_crop[y:y+h, x:x+w]
        if region_crop.size == 0:
            continue
        color_std = np.std(region_crop)
        if color_std < 15:
            continue
        region_edges = edges[y:y+h, x:x+w]
        edge_density = np.count_nonzero(region_edges) / (w * h) if (w * h) > 0 else 0
        if 0.02 < edge_density < 0.35:
            image_regions.append(Region(x=body_region.x + x, y=body_region.y + y, width=w, height=h))

    image_regions = _merge_overlapping_regions(image_regions)
    logger.debug(f"  Detected {len(image_regions)} image regions")
    return image_regions


def _merge_overlapping_regions(regions: list[Region]) -> list[Region]:
    if len(regions) <= 1:
        return regions
    merged = []
    used = [False] * len(regions)
    for i in range(len(regions)):
        if used[i]:
            continue
        current = regions[i]
        cx1, cy1 = current.x, current.y
        cx2, cy2 = current.x2, current.y2
        for j in range(i + 1, len(regions)):
            if used[j]:
                continue
            other = regions[j]
            ox1, oy1 = other.x, other.y
            ox2, oy2 = other.x2, other.y2
            overlap_x = max(0, min(cx2, ox2) - max(cx1, ox1))
            overlap_y = max(0, min(cy2, oy2) - max(cy1, oy1))
            overlap_area = overlap_x * overlap_y
            min_area = min(current.area, other.area)
            if min_area > 0 and overlap_area / min_area > 0.3:
                cx1 = min(cx1, ox1)
                cy1 = min(cy1, oy1)
                cx2 = max(cx2, ox2)
                cy2 = max(cy2, oy2)
                used[j] = True
        merged.append(Region(x=cx1, y=cy1, width=cx2 - cx1, height=cy2 - cy1))
    return merged


def _has_text_content(frame: np.ndarray, region: Region) -> bool:
    crop = region.crop_from(frame)
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    row_means = np.mean(gray, axis=1)
    row_std = np.std(row_means)
    return row_std > 10


def analyze_layout(frame: np.ndarray) -> PageLayout:
    """
    Analyze a frame to identify document layout regions:
    header, footer, body, and embedded images.
    """
    h, w = frame.shape[:2]
    logger.debug(f"Analyzing layout for frame {w}x{h}")

    content = _detect_content_bounds(frame)
    header_height = int(content.height * HEADER_FRACTION)
    footer_height = int(content.height * FOOTER_FRACTION)

    header_candidate = Region(x=content.x, y=content.y, width=content.width, height=header_height)
    footer_candidate = Region(x=content.x, y=content.y + content.height - footer_height,
                              width=content.width, height=footer_height)

    header_region = header_candidate if _has_text_content(frame, header_candidate) else None
    footer_region = footer_candidate if _has_text_content(frame, footer_candidate) else None

    body_top = content.y + (header_height if header_region else 0)
    body_bottom = content.y + content.height - (footer_height if footer_region else 0)
    body_region = Region(x=content.x, y=body_top, width=content.width, height=max(0, body_bottom - body_top))

    image_regions = _detect_image_regions(frame, body_region)

    layout = PageLayout(header_region=header_region, footer_region=footer_region,
                        body_region=body_region, image_regions=image_regions,
                        frame_height=h, frame_width=w)

    logger.debug(f"  Layout: header={'yes' if header_region else 'no'}, "
                 f"footer={'yes' if footer_region else 'no'}, "
                 f"body={body_region.width}x{body_region.height}, "
                 f"images={len(image_regions)}")
    return layout
