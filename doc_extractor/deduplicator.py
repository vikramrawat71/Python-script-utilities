"""
Deduplicator — Filters out near-duplicate frames using perceptual hashing.

Compares consecutive frames using pHash (perceptual hash) and drops those
that are too similar, keeping only frames that represent meaningful
content changes (e.g. a new page, a significant scroll position).
"""

import logging
import cv2
import numpy as np
import imagehash
from PIL import Image

logger = logging.getLogger(__name__)


def _frame_to_phash(frame: np.ndarray) -> imagehash.ImageHash:
    """Convert an OpenCV frame (BGR) to a perceptual hash."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    return imagehash.phash(pil_image)


def deduplicate_frames(
    frames: list[np.ndarray],
    threshold: int = 5
) -> list[np.ndarray]:
    """
    Remove near-duplicate consecutive frames using perceptual hashing.

    Two frames are considered duplicates if the Hamming distance between
    their perceptual hashes is less than or equal to `threshold`.

    Args:
        frames: List of frames (BGR numpy arrays) in sequential order.
        threshold: Maximum Hamming distance to consider frames as duplicates.
                   Lower = stricter (fewer frames kept).
                   Higher = more lenient (more frames kept).

    Returns:
        Filtered list of unique frames.
    """
    if not frames:
        return []

    if len(frames) == 1:
        return frames[:]

    logger.info(f"Deduplicating {len(frames)} frames (threshold={threshold})")

    unique_frames = [frames[0]]
    prev_hash = _frame_to_phash(frames[0])

    for i, frame in enumerate(frames[1:], start=1):
        curr_hash = _frame_to_phash(frame)
        distance = prev_hash - curr_hash

        if distance > threshold:
            unique_frames.append(frame)
            logger.debug(f"  Frame {i}: KEEP (distance={distance})")
        else:
            logger.debug(f"  Frame {i}: DROP (distance={distance})")

        prev_hash = curr_hash

    logger.info(
        f"  Kept {len(unique_frames)} unique frames "
        f"(removed {len(frames) - len(unique_frames)} duplicates)"
    )

    return unique_frames


def deduplicate_with_global_check(
    frames: list[np.ndarray],
    threshold: int = 5
) -> list[np.ndarray]:
    """
    Remove duplicate frames using global comparison (not just consecutive).

    This is more thorough but slower — each frame is compared against all
    previously accepted frames. Useful when the recording scrolls back and
    forth over the same content.

    Args:
        frames: List of frames (BGR numpy arrays).
        threshold: Maximum Hamming distance for duplicate detection.

    Returns:
        Filtered list of unique frames.
    """
    if not frames:
        return []

    logger.info(f"Global deduplication of {len(frames)} frames (threshold={threshold})")

    unique_frames = [frames[0]]
    unique_hashes = [_frame_to_phash(frames[0])]

    for i, frame in enumerate(frames[1:], start=1):
        curr_hash = _frame_to_phash(frame)

        is_duplicate = any(
            (curr_hash - h) <= threshold for h in unique_hashes
        )

        if not is_duplicate:
            unique_frames.append(frame)
            unique_hashes.append(curr_hash)
            logger.debug(f"  Frame {i}: KEEP (unique)")
        else:
            logger.debug(f"  Frame {i}: DROP (duplicate)")

    logger.info(
        f"  Kept {len(unique_frames)} unique frames "
        f"(removed {len(frames) - len(unique_frames)} duplicates)"
    )

    return unique_frames
