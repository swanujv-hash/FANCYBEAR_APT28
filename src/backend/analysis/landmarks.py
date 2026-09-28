from pathlib import Path
import cv2
import numpy as np

_FACE = Path(getattr(getattr(cv2, 'data', None), 'haarcascades', '')) / 'haarcascade_frontalface_default.xml'
_EYE = Path(getattr(getattr(cv2, 'data', None), 'haarcascades', '')) / 'haarcascade_eye.xml'
_SMILE = Path(getattr(getattr(cv2, 'data', None), 'haarcascades', '')) / 'haarcascade_smile.xml'


def _detector(path: Path):
    """Load a Haar detector defensively. Some broken/mixed OpenCV installs expose a partial cv2 module."""
    ctor = getattr(cv2, 'CascadeClassifier', None)
    if ctor is None or not path or not path.exists():
        return None
    try:
        detector = ctor(str(path))
        return detector if not detector.empty() else None
    except Exception:
        return None


def _face_metrics(image, face_detector, eye_detector, smile_detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    if face_detector is None:
        return None
    detected = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(detected) == 0:
        return None
    x, y, w, h = max(detected, key=lambda r: r[2] * r[3])
    roi = gray[y:y+h, x:x+w]
    eyes = eye_detector.detectMultiScale(roi, 1.1, 5, minSize=(12, 12)) if eye_detector is not None and roi.size else []
    mouth_ratio = 0.0
    if smile_detector is not None and roi.size:
        lower = roi[int(h * 0.48):, :]
        mouths = smile_detector.detectMultiScale(lower, 1.5, 15, minSize=(18, 10))
        if len(mouths):
            _, _, _, mh = max(mouths, key=lambda r: r[2] * r[3])
            mouth_ratio = float(mh / max(h, 1))
    return {
        'box': (int(x), int(y), int(w), int(h)),
        'area': int(w * h),
        'center': (x + w / 2.0, y + h / 2.0),
        'eye_count': min(len(eyes), 2),
        'mouth_opening_proxy': mouth_ratio,
    }


def _image_analysis(path: Path, face_detector, eye_detector, smile_detector) -> dict:
    image = cv2.imread(str(path))
    if image is None:
        return {'status': 'error', 'observation': 'The image could not be decoded.', 'significance': 'Facial analysis was not completed.', 'confidence': 0, 'limitations': 'Use a supported JPEG, PNG, WEBP, BMP or TIFF image.', 'frames_analyzed': 0}
    metrics = _face_metrics(image, face_detector, eye_detector, smile_detector)
    if metrics is None:
        reason = 'OpenCV face detection is unavailable in this Python environment.' if face_detector is None else 'No face was detected in the supplied image.'
        return {'status': 'review', 'observation': reason, 'significance': 'No reliable facial geometry could be measured from this image.', 'confidence': 10, 'limitations': 'Face detection depends on image quality, pose, lighting and a healthy OpenCV installation.', 'frames_analyzed': 1, 'frames_with_face': 0, 'method': 'OpenCV Haar face/eye/smile geometry'}
    h, w = image.shape[:2]
    x, y, fw, fh = metrics['box']
    return {
        'status': 'normal',
        'observation': 'A face was detected and facial feature geometry was measurable in the supplied image.',
        'significance': 'The measurements support visual consistency review; they are not a standalone deepfake detector.',
        'confidence': 75,
        'limitations': 'This is feature-box geometry, not identity recognition or proof of manipulation.',
        'method': 'OpenCV Haar face + eye + smile geometry',
        'frames_analyzed': 1,
        'frames_with_face': 1,
        'face_detection_rate': 1.0,
        'image_width_pixels': w,
        'image_height_pixels': h,
        'face_box_x': x,
        'face_box_y': y,
        'face_box_width': fw,
        'face_box_height': fh,
        'face_area_ratio': round(metrics['area'] / max(w * h, 1), 5),
        'eyes_detected': metrics['eye_count'],
        'mouth_opening_proxy': round(metrics['mouth_opening_proxy'], 5),
    }


def analyze_landmarks(path: Path, sample_every: int = 5, max_frames: int = 240) -> dict:
    face_detector = _detector(_FACE)
    eye_detector = _detector(_EYE)
    smile_detector = _detector(_SMILE)

    if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}:
        return _image_analysis(path, face_detector, eye_detector, smile_detector)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {'status': 'error', 'observation': 'The video could not be decoded for facial analysis.', 'significance': 'No facial geometry was measured.', 'confidence': 0, 'limitations': 'Use a supported video codec.', 'frames_analyzed': 0}
    if face_detector is None:
        cap.release()
        return {'status': 'review', 'observation': 'OpenCV face detection is unavailable in this Python environment.', 'significance': 'Facial geometry could not be measured.', 'confidence': 0, 'limitations': 'Restart with START_DEEPTRACE.bat so the launcher can repair the OpenCV installation.', 'frames_analyzed': 0, 'method': 'OpenCV Haar face/eye/smile geometry'}

    faces_seen = 0; face_sizes = []; face_centers = []; eye_counts = []; mouth_values = []; frames_with_face = 0; frames = 0; frame_idx = 0
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok: break
        if frame_idx % sample_every != 0:
            frame_idx += 1; continue
        frames += 1; frame_idx += 1
        metrics = _face_metrics(frame, face_detector, eye_detector, smile_detector)
        if metrics is None: continue
        frames_with_face += 1; faces_seen += 1
        face_sizes.append(metrics['area']); face_centers.append(metrics['center']); eye_counts.append(metrics['eye_count']); mouth_values.append(metrics['mouth_opening_proxy'])
    cap.release()

    if faces_seen == 0:
        return {'status': 'review', 'observation': 'No stable face was detected in the sampled frames.', 'significance': 'Facial geometry and mouth-motion measurements are unavailable.', 'confidence': 15, 'limitations': 'The detector is sensitive to profile views, occlusion, low resolution, masks and extreme lighting.', 'frames_analyzed': frames, 'frames_with_face': 0, 'method': 'OpenCV Haar face/eye/smile geometry'}
    centers = np.array(face_centers); center_jitter = float(np.mean(np.linalg.norm(np.diff(centers, axis=0), axis=1))) if len(centers) > 1 else 0.0
    size_cv = float(np.std(face_sizes) / (np.mean(face_sizes) + 1e-6)); face_rate = frames_with_face / max(frames, 1); eye_rate = float(np.mean(np.array(eye_counts) >= 1)) if eye_counts else 0.0
    return {
        'status': 'normal' if face_rate >= .75 else 'review',
        'observation': 'A consistent face was detected and feature geometry was measurable across the sampled frames.' if face_rate >= .75 else 'A face was detected intermittently; pose, occlusion or image quality reduced feature coverage.',
        'significance': 'The measurements support frame-to-frame consistency review but are not a standalone deepfake detector.',
        'confidence': round(min(90, 35 + 55 * face_rate), 1),
        'limitations': 'Current build uses OpenCV Haar feature geometry (face/eyes/mouth proxy), not a 68/468-point landmark model.',
        'method': 'OpenCV Haar face + eye + smile geometry',
        'frames_analyzed': frames, 'frames_with_face': frames_with_face, 'face_detection_rate': round(face_rate, 4),
        'mean_face_area_pixels': round(float(np.mean(face_sizes)), 2), 'face_area_variation': round(size_cv, 4),
        'mean_face_center_motion_pixels': round(center_jitter, 3), 'frames_with_eyes_detected': round(eye_rate, 4),
        'mean_mouth_opening_proxy': round(float(np.mean(mouth_values)), 5), 'mouth_motion_variation': round(float(np.std(mouth_values)), 5),
    }
