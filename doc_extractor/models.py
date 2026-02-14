"""
Data models used across the document extraction pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from PIL import Image


@dataclass
class Region:
    """A rectangular region within a frame, defined by pixel coordinates."""
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def crop_from(self, frame: np.ndarray) -> np.ndarray:
        """Crop this region from a frame (numpy array in BGR/RGB format)."""
        return frame[self.y:self.y2, self.x:self.x2]


@dataclass
class TextBlock:
    """A block of text extracted from a region, with formatting hints."""
    text: str
    font_size_estimate: float = 12.0
    is_bold: bool = False
    is_heading: bool = False
    line_number: int = 0
    confidence: float = 0.0


@dataclass
class TextResult:
    """Result of OCR on a region, containing multiple text blocks."""
    blocks: list[TextBlock] = field(default_factory=list)
    raw_text: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.raw_text.strip()


@dataclass
class PageLayout:
    """Layout analysis result for a single frame/page."""
    header_region: Optional[Region] = None
    footer_region: Optional[Region] = None
    body_region: Optional[Region] = None
    image_regions: list[Region] = field(default_factory=list)
    frame_height: int = 0
    frame_width: int = 0


@dataclass
class PageContent:
    """Extracted content for a single page."""
    header_text: Optional[TextResult] = None
    footer_text: Optional[TextResult] = None
    body_text: Optional[TextResult] = None
    images: list[Image.Image] = field(default_factory=list)
    page_number: int = 0


@dataclass
class DocumentContent:
    """Full document content extracted from all pages."""
    pages: list[PageContent] = field(default_factory=list)
    common_header: Optional[TextResult] = None
    common_footer: Optional[TextResult] = None

    @property
    def page_count(self) -> int:
        return len(self.pages)
