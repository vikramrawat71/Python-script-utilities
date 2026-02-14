# Screen Recording Document Extractor

A Python CLI utility that reads a screen recording of a document (e.g., scrolling PDF, Word doc, or web page) and extracts its content — **headers, footers, body text, and images** — into a formatted `.docx` file that closely mirrors the original layout.

## Features

- **Video Processing** — Reads `.mp4`, `.avi`, `.mkv`, and other OpenCV-supported formats
- **Smart Frame Deduplication** — Uses perceptual hashing to keep only unique page snapshots
- **Layout Detection** — Identifies header, footer, body, and embedded image regions using OpenCV contour analysis
- **Dual OCR Engines** — Supports Tesseract (fast) and EasyOCR (deep learning, higher accuracy)
- **Formatting Preservation** — Detects headings, paragraphs, bold text, and approximate font sizes
- **Image Extraction** — Crops and embeds document images into the output
- **Clean .docx Output** — Generates a Word document with proper headers, footers, styled text, and inline images

## Architecture

```
Screen Recording → Frame Extraction → Deduplication → Layout Analysis → OCR + Image Extraction → DOCX Output
```

| Stage | Technology | Description |
|-------|-----------|-------------|
| Frame Extraction | OpenCV | Reads video, samples frames at configurable interval |
| Deduplication | imagehash | Perceptual hashing removes near-duplicate frames |
| Layout Analysis | OpenCV | Contour detection + heuristics for header/footer/body/images |
| OCR | Tesseract / EasyOCR | Text extraction with paragraph & heading detection |
| Image Extraction | OpenCV + Pillow | Crops image regions, applies enhancement |
| DOCX Generation | python-docx | Assembles formatted Word document |

## Prerequisites

### Python 3.10+

### Tesseract OCR (required for default engine)
- **Windows**: Download from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr`

After installation, ensure `tesseract` is on your PATH, or set the path in your environment:
```bash
# Windows example (if not on PATH)
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Installation

```bash
# Clone the repository
git clone https://github.com/vikramrawat71/Python-script-utilities.git
cd Python-script-utilities

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py -i recording.mp4 -o output.docx
```

### With EasyOCR (better accuracy)
```bash
python main.py -i recording.mp4 -o output.docx --ocr-engine easyocr
```

### Verbose mode with custom settings
```bash
python main.py -i recording.mp4 -o output.docx --interval 1.0 --threshold 3 -v
```

### All Options
```
usage: main.py [-h] -i INPUT -o OUTPUT [--interval INTERVAL]
               [--ocr-engine {tesseract,easyocr}] [--threshold THRESHOLD]
               [--global-dedup] [-v]

Options:
  -i, --input          Path to screen recording video (.mp4, .avi, .mkv)
  -o, --output         Path for output .docx file
  --interval           Frame sampling interval in seconds (default: 0.5)
  --ocr-engine         OCR engine: "tesseract" or "easyocr" (default: tesseract)
  --threshold          Deduplication hash threshold (default: 5, lower=stricter)
  --global-dedup       Use global dedup (handles scroll-back, slower)
  -v, --verbose        Enable detailed debug logging
```

## Tips for Best Results

1. **High resolution** — Record at 1080p or higher for best OCR accuracy
2. **Slow, steady scrolling** — Scroll the document slowly to avoid motion blur
3. **Good contrast** — Ensure the document has clear text against its background
4. **Full-screen** — Recording the document in full-screen reduces window chrome artifacts
5. **Adjust threshold** — Lower `--threshold` to keep more frames, higher to be more aggressive with dedup

## Project Structure

```
├── main.py                      # CLI entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── doc_extractor/
    ├── __init__.py
    ├── models.py                # Data models (Region, TextBlock, PageLayout, etc.)
    ├── frame_extractor.py       # Video frame extraction (OpenCV)
    ├── deduplicator.py          # Perceptual hash deduplication
    ├── layout_analyzer.py       # Document layout detection
    ├── ocr_engine.py            # OCR (Tesseract + EasyOCR)
    ├── image_extractor.py       # Image region extraction
    └── docx_builder.py          # Word document generation
```

## License

MIT
