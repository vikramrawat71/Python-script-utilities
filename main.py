"""
Screen Recording Document Extractor — CLI Entry Point

Usage:
    python main.py -i recording.mp4 -o output.docx
    python main.py -i recording.mp4 -o output.docx --ocr-engine easyocr -v
"""

import argparse
import logging
import os
import sys
import tempfile
import time

from doc_extractor.frame_extractor import extract_frames, get_video_info
from doc_extractor.deduplicator import deduplicate_frames, deduplicate_with_global_check
from doc_extractor.layout_analyzer import analyze_layout
from doc_extractor.ocr_engine import extract_text
from doc_extractor.image_extractor import extract_images
from doc_extractor.docx_builder import build_document
from doc_extractor.models import PageContent, DocumentContent, TextResult


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")


def _find_common_text(text_results: list[TextResult | None]) -> TextResult | None:
    valid = [t for t in text_results if t and not t.is_empty]
    if not valid:
        return None
    texts = [t.raw_text.strip() for t in valid]
    if not texts:
        return None
    from collections import Counter
    counter = Counter(texts)
    most_common_text, count = counter.most_common(1)[0]
    if count > len(text_results) / 2:
        for t in valid:
            if t.raw_text.strip() == most_common_text:
                return t
    return None


def run_pipeline(input_path: str, output_path: str, interval: float = 0.5,
                 ocr_engine: str = "tesseract", threshold: int = 5, global_dedup: bool = False):
    logger = logging.getLogger("pipeline")
    total_start = time.time()

    logger.info("=" * 60)
    logger.info("STAGE 1: Reading video information")
    logger.info("=" * 60)
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    info = get_video_info(input_path)
    logger.info(f"  Resolution: {info['width']}x{info['height']}")
    logger.info(f"  Duration: {info['duration']:.1f}s, FPS: {info['fps']:.1f}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2: Extracting frames")
    logger.info("=" * 60)
    t = time.time()
    frames = extract_frames(input_path, interval_sec=interval)
    logger.info(f"  Completed in {time.time() - t:.1f}s")
    if not frames:
        logger.error("No frames extracted from the video!")
        sys.exit(1)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 3: Removing duplicate frames")
    logger.info("=" * 60)
    t = time.time()
    if global_dedup:
        unique_frames = deduplicate_with_global_check(frames, threshold=threshold)
    else:
        unique_frames = deduplicate_frames(frames, threshold=threshold)
    logger.info(f"  Completed in {time.time() - t:.1f}s")
    if not unique_frames:
        logger.error("All frames were filtered as duplicates. Try a lower threshold.")
        sys.exit(1)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 4: Analyzing layout, extracting text & images")
    logger.info("=" * 60)
    t = time.time()
    temp_dir = tempfile.mkdtemp(prefix="doc_extractor_")
    pages: list[PageContent] = []
    for idx, frame in enumerate(unique_frames):
        logger.info(f"  Processing page {idx + 1}/{len(unique_frames)}")
        layout = analyze_layout(frame)
        header_text = None
        footer_text = None
        body_text = None
        if layout.header_region:
            header_crop = layout.header_region.crop_from(frame)
            header_text = extract_text(header_crop, engine=ocr_engine)
        if layout.footer_region:
            footer_crop = layout.footer_region.crop_from(frame)
            footer_text = extract_text(footer_crop, engine=ocr_engine)
        if layout.body_region:
            body_crop = layout.body_region.crop_from(frame)
            body_text = extract_text(body_crop, engine=ocr_engine)
        images = extract_images(frame, layout.image_regions,
                                output_dir=os.path.join(temp_dir, f"page_{idx:03d}"))
        pages.append(PageContent(header_text=header_text, footer_text=footer_text,
                                 body_text=body_text, images=images, page_number=idx + 1))
    logger.info(f"  Completed in {time.time() - t:.1f}s")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 5: Identifying common headers/footers")
    logger.info("=" * 60)
    common_header = _find_common_text([p.header_text for p in pages])
    common_footer = _find_common_text([p.footer_text for p in pages])
    if common_header:
        logger.info(f"  Common header detected: {common_header.raw_text[:60]}...")
    if common_footer:
        logger.info(f"  Common footer detected: {common_footer.raw_text[:60]}...")
    doc_content = DocumentContent(pages=pages, common_header=common_header, common_footer=common_footer)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 6: Building .docx document")
    logger.info("=" * 60)
    t = time.time()
    build_document(doc_content, output_path)
    logger.info(f"  Completed in {time.time() - t:.1f}s")

    total_time = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Pages extracted: {len(pages)}")
    logger.info(f"  Total images:    {sum(len(p.images) for p in pages)}")
    logger.info(f"  Output file:     {os.path.abspath(output_path)}")
    logger.info(f"  Total time:      {total_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Extract document content from a screen recording and save as .docx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python main.py -i recording.mp4 -o output.docx\n  python main.py -i recording.mp4 -o output.docx --ocr-engine easyocr -v")
    parser.add_argument("-i", "--input", required=True, help="Path to screen recording video")
    parser.add_argument("-o", "--output", required=True, help="Path for output .docx file")
    parser.add_argument("--interval", type=float, default=0.5, help="Frame sampling interval in seconds (default: 0.5)")
    parser.add_argument("--ocr-engine", choices=["tesseract", "easyocr"], default="tesseract", help="OCR engine (default: tesseract)")
    parser.add_argument("--threshold", type=int, default=5, help="Dedup hash threshold (default: 5)")
    parser.add_argument("--global-dedup", action="store_true", help="Use global deduplication")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    setup_logging(args.verbose)
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    output = args.output
    if not output.lower().endswith(".docx"):
        output += ".docx"
    run_pipeline(input_path=args.input, output_path=output, interval=args.interval,
                 ocr_engine=args.ocr_engine, threshold=args.threshold, global_dedup=args.global_dedup)


if __name__ == "__main__":
    main()
