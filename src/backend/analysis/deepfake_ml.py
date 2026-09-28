"""Machine-learning deepfake screening for images and sampled video frames.

Uses the corrected Community Forensics ViT-Small ONNX model. The model is a
screening classifier for AI-generated imagery; video is handled by sampling
frames and aggregating their frame-level scores. It is not a legal or
calibrated forensic certainty.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception:
    ort = None

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "community_forensics_v11_fp32.onnx"
MODEL_URL = (
    "https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/"
    "resolve/main/onnx/model.onnx"
)
MIN_MODEL_SIZE = 75_000_000
_SESSION = None


def _download_model() -> None:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MIN_MODEL_SIZE:
        return
    tmp = MODEL_PATH.with_suffix(".download")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if tmp.stat().st_size < MIN_MODEL_SIZE:
            raise RuntimeError("The downloaded detector model is incomplete.")
        tmp.replace(MODEL_PATH)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "The ML detector model is not cached. Internet access is required "
            "once to download the detector (~84 MB)."
        ) from exc


def _session():
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    if ort is None:
        raise RuntimeError("onnxruntime is not installed.")
    _download_model()
    _SESSION = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _SESSION


def _largest_face(image_bgr: np.ndarray):
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return None
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(64, 64))
        if len(faces) == 0:
            return None
        return max(faces, key=lambda b: int(b[2]) * int(b[3]))
    except Exception:
        return None


def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # Match the current Community Forensics preprocessing: shortest edge -> 440,
    # preserve aspect ratio, then center crop to 384x384 and CLIP-normalize.
    scale = 440.0 / min(h, w)
    nh, nw = max(440, int(round(h * scale))), max(440, int(round(w * scale)))
    img = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BICUBIC)
    left = max(0, (nw - 384) // 2)
    top = max(0, (nh - 384) // 2)
    img = img.crop((left, top, left + 384, top + 384))
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    return np.expand_dims(arr.transpose(2, 0, 1), 0).astype(np.float32)


def _predict_frame(image_bgr: np.ndarray) -> tuple[float, bool]:
    session = _session()
    input_name = session.get_inputs()[0].name
    output = np.asarray(session.run(None, {input_name: _preprocess(image_bgr)})[0]).reshape(-1)
    # Current model is single-logit: sigmoid(logit) is its fake score.
    logit = float(output[0])
    fake_score = 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))
    return float(fake_score), _largest_face(image_bgr) is not None


def _status(score: float) -> str:
    if score >= 0.80:
        return "possible_anomaly"
    if score >= 0.55:
        return "review"
    return "normal"


def analyze_image_ml(path: Path) -> dict:
    try:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError("Unable to decode the image.")
        score, face_found = _predict_frame(image)
        return {
            "status": _status(score),
            "observation": (
                "The ML detector produced a strong synthetic-media signal."
                if score >= 0.80 else
                "The ML detector produced a review-level synthetic-media signal."
                if score >= 0.55 else
                "The ML detector produced a lower synthetic-media signal."
            ),
            "model": "Community Forensics ViT-Small v1.1 (ONNX FP32)",
            "model_score": round(score * 100, 2),
            "verdict": "possible_deepfake" if score >= 0.55 else "no_strong_ml_signal",
            "frames_evaluated": 1,
            "face_detected": face_found,
            "confidence": round(max(score, 1 - score) * 100, 2),
            "significance": "This is the primary ML screening signal. It is not a calibrated legal/forensic probability and does not by itself prove manipulation.",
            "limitations": "Performance can vary on unseen generators, edited/recompressed media, unusual subjects, and non-photographic content. Corroborate with independent forensic evidence.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "observation": "ML deepfake screening could not be completed.",
            "error": str(exc),
            "confidence": None,
            "significance": "No ML authenticity conclusion was made.",
            "limitations": "Install onnxruntime and allow the detector model to download on first use. The model output is probabilistic and should not be treated as proof of authenticity.",
        }


def analyze_video_ml(path: Path, max_frames: int = 12) -> dict:
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError("Unable to open the video.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_count = min(max_frames, max(total, 1))
        indices = np.linspace(0, max(total - 1, 0), num=sample_count, dtype=int)
        scores = []
        face_frames = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            score, face_found = _predict_frame(frame)
            scores.append(score)
            face_frames += int(face_found)
        cap.release()
        if not scores:
            raise RuntimeError("No decodable video frames were available.")

        arr = np.asarray(scores, dtype=float)
        mean_score = float(np.mean(arr))
        median_score = float(np.median(arr))
        high_fraction = float(np.mean(arr >= 0.55))
        strong_fraction = float(np.mean(arr >= 0.80))

        # Repeated evidence is required for a video-level flag; one anomalous frame
        # is reported as review rather than being treated as a verdict.
        if strong_fraction >= 0.5 and mean_score >= 0.70:
            status = "possible_anomaly"
        elif high_fraction >= 0.5 and mean_score >= 0.55:
            status = "review"
        else:
            status = "normal"

        return {
            "status": status,
            "observation": (
                "Multiple sampled frames produced a strong repeated synthetic-media signal."
                if status == "possible_anomaly" else
                "Multiple sampled frames produced a review-level synthetic-media signal."
                if status == "review" else
                "Sampled frames did not produce a strong repeated synthetic-media signal."
            ),
            "model": "Community Forensics ViT-Small v1.1 (ONNX FP32)",
            "model_score": round(mean_score * 100, 2),
            "median_frame_score": round(median_score * 100, 2),
            "max_frame_score": round(float(np.max(arr)) * 100, 2),
            "high_signal_frame_fraction": round(high_fraction * 100, 2),
            "strong_signal_frame_fraction": round(strong_fraction * 100, 2),
            "frames_evaluated": len(scores),
            "face_detected_frame_fraction": round((face_frames / len(scores)) * 100, 2),
            "confidence": round(max(mean_score, 1 - mean_score) * 100, 2),
            "significance": "Repeated frame-level ML evidence is stronger than a single outlier, but the score is not a calibrated legal/forensic probability.",
            "limitations": "The ML component is an image detector applied to sampled video frames; it does not independently establish temporal manipulation, identity, or legal authenticity.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "observation": "ML video deepfake screening could not be completed.",
            "error": str(exc),
            "confidence": None,
            "significance": "No ML authenticity conclusion was made.",
            "limitations": "Install onnxruntime and allow the detector model to download on first use. The model output is probabilistic and should not be treated as proof of authenticity.",
        }
