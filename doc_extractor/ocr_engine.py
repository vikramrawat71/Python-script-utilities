"""
OCR Engine — Extracts text from image regions using Tesseract or EasyOCR.

Provides paragraph detection, font size estimation, and heading detection
based on bounding box heights from OCR output.
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

from .models import TextBlock, TextResult


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened


def _estimate_font_size(bbox_height: float, image_height: int) -> float:
    return max(6.0, bbox_height / 1.33)


def _detect_paragraphs(lines: list[dict], line_spacing_threshold: float = 1.8) -> list[list[dict]]:
    if not lines:
        return []
    if len(lines) == 1:
        return [lines]
    sorted_lines = sorted(lines, key=lambda l: l["y"])
    heights = [l["height"] for l in sorted_lines if l["height"] > 0]
    avg_height = np.mean(heights) if heights else 20
    paragraphs = []
    current_para = [sorted_lines[0]]
    for i in range(1, len(sorted_lines)):
        prev = sorted_lines[i - 1]
        curr = sorted_lines[i]
        gap = curr["y"] - (prev["y"] + prev["height"])
        if gap > avg_height * line_spacing_threshold:
            paragraphs.append(current_para)
            current_para = [curr]
        else:
            current_para.append(curr)
    if current_para:
        paragraphs.append(current_para)
    return paragraphs


def extract_text_tesseract(image: np.ndarray) -> TextResult:
    try:
        import pytesseract
    except ImportError:
        logger.error("pytesseract not installed. Run: pip install pytesseract")
        return TextResult(raw_text="", blocks=[])

    processed = _preprocess_for_ocr(image)
    h_img = image.shape[0]

    try:
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, config="--psm 6")
    except Exception as e:
        logger.warning(f"Tesseract OCR failed: {e}")
        try:
            raw = pytesseract.image_to_string(processed, config="--psm 6")
            return TextResult(raw_text=raw.strip(), blocks=[TextBlock(text=raw.strip(), confidence=0)])
        except Exception as e2:
            logger.error(f"Tesseract fallback also failed: {e2}")
            return TextResult(raw_text="", blocks=[])

    lines_data = {}
    n_items = len(data["text"])
    for i in range(n_items):
        text = data["text"][i].strip()
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
        if not text or conf < 30:
            continue
        line_num = data["line_num"][i]
        block_num = data["block_num"][i]
        key = (block_num, line_num)
        if key not in lines_data:
            lines_data[key] = {"text": [], "y": data["top"][i], "height": data["height"][i],
                               "x": data["left"][i], "width": data["width"][i], "conf": []}
        lines_data[key]["text"].append(text)
        lines_data[key]["conf"].append(conf)
        lines_data[key]["y"] = min(lines_data[key]["y"], data["top"][i])
        lines_data[key]["height"] = max(lines_data[key]["height"],
                                        data["top"][i] + data["height"][i] - lines_data[key]["y"])
        right = data["left"][i] + data["width"][i]
        if right > lines_data[key]["x"] + lines_data[key]["width"]:
            lines_data[key]["width"] = right - lines_data[key]["x"]

    lines = []
    for key in sorted(lines_data.keys()):
        ld = lines_data[key]
        lines.append({"text": " ".join(ld["text"]), "y": ld["y"], "height": ld["height"],
                      "x": ld["x"], "width": ld["width"], "conf": np.mean(ld["conf"])})

    paragraphs = _detect_paragraphs(lines)
    blocks = []
    all_heights = [l["height"] for l in lines if l["height"] > 0]
    median_height = float(np.median(all_heights)) if all_heights else 20.0

    for para_idx, para_lines in enumerate(paragraphs):
        para_text = " ".join(l["text"] for l in para_lines)
        avg_height = np.mean([l["height"] for l in para_lines])
        avg_conf = np.mean([l["conf"] for l in para_lines])
        font_size = _estimate_font_size(avg_height, h_img)
        is_heading = avg_height > median_height * 1.3
        is_bold = avg_height > median_height * 1.2
        blocks.append(TextBlock(text=para_text, font_size_estimate=round(font_size, 1),
                                is_bold=is_bold, is_heading=is_heading,
                                line_number=para_idx, confidence=round(avg_conf, 1)))

    raw_text = "\n\n".join(b.text for b in blocks)
    return TextResult(blocks=blocks, raw_text=raw_text)


def extract_text_easyocr(image: np.ndarray) -> TextResult:
    try:
        import easyocr
    except ImportError:
        logger.error("easyocr not installed. Run: pip install easyocr")
        return TextResult(raw_text="", blocks=[])

    processed = _preprocess_for_ocr(image)
    h_img = image.shape[0]

    try:
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(processed)
    except Exception as e:
        logger.error(f"EasyOCR failed: {e}")
        return TextResult(raw_text="", blocks=[])

    if not results:
        return TextResult(raw_text="", blocks=[])

    lines = []
    for (bbox, text, conf) in results:
        if conf < 0.3 or not text.strip():
            continue
        points = np.array(bbox)
        x = int(np.min(points[:, 0]))
        y = int(np.min(points[:, 1]))
        x2 = int(np.max(points[:, 0]))
        y2 = int(np.max(points[:, 1]))
        lines.append({"text": text.strip(), "y": y, "height": y2 - y,
                      "x": x, "width": x2 - x, "conf": conf * 100})

    lines.sort(key=lambda l: l["y"])
    paragraphs = _detect_paragraphs(lines)
    blocks = []
    all_heights = [l["height"] for l in lines if l["height"] > 0]
    median_height = float(np.median(all_heights)) if all_heights else 20.0

    for para_idx, para_lines in enumerate(paragraphs):
        para_text = " ".join(l["text"] for l in para_lines)
        avg_height = np.mean([l["height"] for l in para_lines])
        avg_conf = np.mean([l["conf"] for l in para_lines])
        font_size = _estimate_font_size(avg_height, h_img)
        is_heading = avg_height > median_height * 1.3
        is_bold = avg_height > median_height * 1.2
        blocks.append(TextBlock(text=para_text, font_size_estimate=round(font_size, 1),
                                is_bold=is_bold, is_heading=is_heading,
                                line_number=para_idx, confidence=round(avg_conf, 1)))

    raw_text = "\n\n".join(b.text for b in blocks)
    return TextResult(blocks=blocks, raw_text=raw_text)


def extract_text(image: np.ndarray, engine: str = "tesseract") -> TextResult:
    if engine == "easyocr":
        return extract_text_easyocr(image)
    else:
        return extract_text_tesseract(image)
