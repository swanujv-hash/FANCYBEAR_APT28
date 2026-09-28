from pathlib import Path
import cv2
import math
import numpy as np


def _blockiness(gray: np.ndarray) -> float:
    """Estimate 8x8 DCT/block boundary energy. Higher can indicate stronger blocking."""
    g = gray.astype(np.float32)
    h, w = g.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 < 16 or w8 < 16:
        return 0.0
    g = g[:h8, :w8]
    v1 = g[:, 8:w8:8]
    v2 = g[:, 7:w8-1:8]
    h1 = g[8:h8:8, :]
    h2 = g[7:h8-1:8, :]
    vertical = np.abs(v1 - v2).mean() if v1.size and v1.shape == v2.shape else 0.0
    horizontal = np.abs(h1 - h2).mean() if h1.size and h1.shape == h2.shape else 0.0
    return float((vertical + horizontal) / 2.0)


def _laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())



def analyze_image(path: Path) -> dict:
    """Compression/quality screening for still images."""
    image = cv2.imread(str(path))
    if image is None:
        return {"status":"error","observation":"The image could not be decoded.","significance":"Image compression measurements could not be calculated.","confidence":0,"limitations":"Use a supported JPEG, PNG, WEBP, BMP or TIFF image."}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (min(640, max(64, w)), min(640, max(64, h))))
    block = _blockiness(gray)
    lap = _laplacian(gray)
    edges = cv2.Canny(small, 80, 160)
    edge_density = float(np.mean(edges > 0))
    # JPEG quantization tables are available through PIL when installed; use them only as a
    # supporting clue and never convert them into a fake/not-fake verdict.
    jpeg_quality = None
    try:
        from PIL import Image
        with Image.open(path) as im:
            q = im.quantization
            if q:
                vals = [v for table in q.values() for v in table]
                avg_q = float(np.mean(vals)) if vals else 0.0
                jpeg_quality = round(max(1, min(100, 100 - (avg_q - 1) * 0.5)), 1)
    except Exception:
        pass
    if block > 7.5:
        status = "review"
        observation = "The image shows measurable 8x8 block-boundary energy consistent with lossy compression."
    else:
        status = "normal"
        observation = "No unusually strong 8x8 block-boundary pattern was measured in the supplied image."
    return {
        "status": status, "observation": observation,
        "significance": "Compression measurements can identify recompression/quality clues but cannot prove manipulation by themselves.",
        "confidence": 55,
        "limitations": "Still-image compression metrics are screening indicators and should be compared with the original/source file when available.",
        "width_pixels": w, "height_pixels": h, "average_8x8_block_boundary_energy": round(block,4),
        "laplacian_variance": round(lap,3), "edge_density": round(edge_density,5),
        "estimated_jpeg_quality": jpeg_quality,
    }


def analyze_frames(path: Path, sample_every: int = 5, max_frames: int = 300) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {
            "status": "error",
            "observation": "The video could not be decoded by OpenCV.",
            "significance": "Compression and frame-quality measurements could not be calculated.",
            "confidence": 0,
            "limitations": "Use a playable MP4/MOV/WebM/AVI file with a supported codec.",
            "frames_analyzed": 0,
        }

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    duration = total_frames / fps if fps > 0 else None

    selected = 0
    block_values, blur_values, brightness_values, edge_values, temporal_values = [], [], [], [], []
    previous = None
    frame_index = 0

    while selected < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (320, 180))
            brightness_values.append(float(gray.mean()))
            blur_values.append(_laplacian(gray))
            block_values.append(_blockiness(gray))
            edges = cv2.Canny(gray_small, 80, 160)
            edge_values.append(float(np.mean(edges > 0)))
            if previous is not None:
                temporal_values.append(float(np.mean(cv2.absdiff(gray_small, previous))))
            previous = gray_small
            selected += 1
        frame_index += 1

    capture.release()

    if selected == 0:
        return {
            "status": "error",
            "observation": "No usable video frames were sampled.",
            "significance": "No compression indicators are available.",
            "confidence": 0,
            "limitations": "The media may be unreadable or unsupported.",
            "frames_analyzed": 0,
        }

    def mean(xs):
        return float(np.mean(xs)) if xs else 0.0
    def std(xs):
        return float(np.std(xs)) if xs else 0.0

    avg_block = mean(block_values)
    block_std = std(block_values)
    avg_blur = mean(blur_values)
    avg_edge = mean(edge_values)
    temporal_mean = mean(temporal_values)

    # These are screening indicators, not a codec detector or deepfake classifier.
    block_z = (block_values[-1] - avg_block) / (block_std + 1e-6) if block_values else 0.0
    variability = std(block_values) / (avg_block + 1e-6)

    if avg_block > 7.5 and variability > 0.25:
        status = "possible_anomaly"
        observation = "The sampled frames show elevated and inconsistent 8x8 boundary energy."
        significance = "Uneven blocking can occur after recompression, editing, resizing, or mixed-source assembly and merits forensic review."
    elif avg_block > 7.5:
        status = "review"
        observation = "Noticeable 8x8 block-boundary energy was measured across sampled frames."
        significance = "Blocking is consistent with lossy compression but does not by itself indicate manipulation."
    else:
        status = "normal"
        observation = "No unusually strong 8x8 blocking pattern was measured in the sampled frames."
        significance = "The measured blocking is compatible with ordinary video compression; this does not establish authenticity."

    return {
        "status": status,
        "observation": observation,
        "significance": significance,
        "confidence": 55,
        "limitations": "Compression metrics are screening indicators. They cannot identify a specific encoder or prove a deepfake without source/comparison evidence.",
        "frames_analyzed": selected,
        "total_frames_reported": total_frames,
        "fps_reported": round(fps, 3),
        "duration_seconds": round(duration, 3) if duration is not None else None,
        "average_8x8_block_boundary_energy": round(avg_block, 4),
        "block_energy_variation": round(block_std, 4),
        "block_energy_relative_variation": round(variability, 4),
        "last_frame_block_z_score": round(block_z, 3),
        "average_laplacian_variance": round(avg_blur, 3),
        "average_edge_density": round(avg_edge, 5),
        "average_frame_difference": round(temporal_mean, 4),
        "sampling_interval_frames": sample_every,
    }
