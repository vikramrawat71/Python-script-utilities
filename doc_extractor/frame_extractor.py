"""
Frame Extractor — Reads a video file and extracts frames at a configurable interval.

Uses OpenCV (cv2) to read the video and sample frames based on the
video's native FPS and the desired sampling interval.
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, interval_sec: float = 0.5) -> list[np.ndarray]:
    """
    Extract frames from a video file at a given time interval.

    Args:
        video_path: Path to the video file (.mp4, .avi, .mkv, etc.)
        interval_sec: Time interval in seconds between sampled frames.

    Returns:
        List of frames as numpy arrays (BGR format).

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"Video: {video_path}")
    logger.info(f"  FPS: {fps:.1f}, Total frames: {total_frames}, Duration: {duration:.1f}s")

    # Calculate frame interval
    frame_interval = max(1, int(fps * interval_sec))
    logger.info(f"  Sampling every {frame_interval} frames ({interval_sec}s interval)")

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frames.append(frame)
            logger.debug(f"  Captured frame {frame_idx} (t={frame_idx/fps:.2f}s)")

        frame_idx += 1

    cap.release()
    logger.info(f"  Extracted {len(frames)} frames from {frame_idx} total")

    return frames


def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video file.

    Returns:
        Dict with keys: fps, total_frames, duration, width, height.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0

    cap.release()
    return info
