from pathlib import Path
import subprocess
import wave
import tempfile
import os
import cv2
import numpy as np

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

_FACE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_SMILE = cv2.data.haarcascades + "haarcascade_smile.xml"


def _extract_audio(path: Path):
    if imageio_ffmpeg is not None:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    else:
        ffmpeg = "ffmpeg"
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [ffmpeg, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", tmp.name]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90, check=True)
        return tmp.name
    except Exception:
        try: os.unlink(tmp.name)
        except OSError: pass
        return None


def _audio_envelope(wav_path: str, hop_ms: int = 40):
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate(); n = wf.getnframes(); raw = wf.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if len(audio) == 0: return None, None
    hop = max(1, int(sr * hop_ms / 1000))
    win = hop * 2
    values=[]; times=[]
    for start in range(0, max(1,len(audio)-win+1), hop):
        chunk=audio[start:start+win]
        rms=float(np.sqrt(np.mean(chunk*chunk))+1e-9)
        values.append(rms); times.append((start+len(chunk)/2)/sr)
    env=np.array(values,dtype=np.float32)
    env=(env-env.mean())/(env.std()+1e-6)
    return np.array(times), env


def analyze_audio_lip_sync(path: Path, sample_every: int = 3, max_frames: int = 360) -> dict:
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        return {"status":"not_applicable","observation":"Audio-lip synchronization is not applicable to a still image.","significance":"The image can be reviewed with metadata, compression and facial-feature analysis instead.","confidence":None,"limitations":"Upload a video with an audio stream to run temporal lip-sync analysis.","frames_analyzed":0}
    cascade = getattr(cv2, "CascadeClassifier", None)
    if cascade is None:
        return {"status":"not_available","observation":"OpenCV face detection is unavailable in this Python environment.","significance":"Lip-sync analysis was not attempted because mouth tracking cannot be built safely.","confidence":0,"limitations":"Restart with START_DEEPTRACE.bat so the launcher can repair the OpenCV installation.","frames_analyzed":0}
    audio_path = _extract_audio(path)
    if not audio_path:
        return {"status":"not_available","observation":"Audio could not be extracted from the video.","significance":"Lip/audio synchronization could not be measured.","confidence":0,"limitations":"The bundled FFmpeg extractor could not decode an audio stream. Try a standard MP4 with AAC audio.","frames_analyzed":0}

    try:
        audio_times, audio_env = _audio_envelope(audio_path)
    finally:
        try: os.unlink(audio_path)
        except OSError: pass
    if audio_times is None or len(audio_env)<5:
        return {"status":"not_available","observation":"The video did not contain a usable audio signal.","significance":"No synchronization measurement was possible.","confidence":0,"limitations":"A continuous audio stream is required for this screening test.","frames_analyzed":0}

    try:
        face_detector=cascade(_FACE)
        smile_detector=cascade(_SMILE)
    except Exception:
        return {"status":"not_available","observation":"OpenCV face detectors could not be initialized.","significance":"Lip-sync analysis was not completed.","confidence":0,"limitations":"Restart with START_DEEPTRACE.bat to repair the OpenCV installation.","frames_analyzed":0}
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"status":"error","observation":"Video could not be decoded for mouth-motion analysis.","significance":"No lip-sync measurement was possible.","confidence":0,"limitations":"Use a supported video codec.","frames_analyzed":0}
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if fps<=0: fps=25.0
    mouth_values=[]; video_times=[]
    idx=0; frames=0
    while frames<max_frames:
        ok,frame=cap.read()
        if not ok: break
        if idx%sample_every: idx+=1; continue
        idx+=1; frames+=1
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_detector.detectMultiScale(gray,1.1,5,minSize=(50,50))
        if len(faces)==0: continue
        x,y,w,h=max(faces,key=lambda r:r[2]*r[3])
        lower=gray[y+int(.48*h):y+h,x:x+w]
        mouth=0.0
        if smile_detector is not None and lower.size:
            ms=smile_detector.detectMultiScale(lower,1.5,20,minSize=(20,10))
            if len(ms):
                mx,my,mw,mh=max(ms,key=lambda r:r[2]*r[3]); mouth=mh/max(h,1)
        mouth_values.append(mouth); video_times.append((idx-1)/fps)
    cap.release()

    if len(mouth_values)<8:
        return {"status":"review","observation":"A usable face/mouth motion track could not be built for enough frames.","significance":"Lip-sync correlation is inconclusive.","confidence":20,"limitations":"Mouth tracking is sensitive to pose, resolution, occlusion and facial expressions.","frames_analyzed":frames,"audio_duration_seconds":round(float(audio_times[-1]),2)}

    vt=np.array(video_times); mv=np.array(mouth_values,dtype=np.float32)
    # Interpolate audio energy onto video timestamps.
    av=np.interp(vt,audio_times,audio_env)
    mv=(mv-mv.mean())/(mv.std()+1e-6)
    # Compare at small temporal offsets; positive lag means audio envelope is shifted later.
    best_corr=-1.0; best_lag=0.0
    for lag in np.arange(-0.60,0.61,0.04):
        shifted=np.interp(vt+lag,audio_times,audio_env,left=np.nan,right=np.nan)
        mask=np.isfinite(shifted)
        if mask.sum()<8: continue
        c=float(np.corrcoef(mv[mask],shifted[mask])[0,1]) if np.std(shifted[mask])>1e-6 else 0.0
        if c>best_corr: best_corr=c; best_lag=float(lag)

    if best_corr>=0.55:
        status="normal"; obs="Mouth-motion and audio-energy patterns show a measurable temporal correlation."
    elif best_corr>=0.25:
        status="review"; obs="Mouth-motion and audio-energy show only a moderate correlation; timing should be reviewed manually."
    else:
        status="possible_anomaly"; obs="The measured mouth-motion/audio-energy correlation is low in the sampled segment."

    return {
        "status":status,
        "observation":obs,
        "significance":"This is a temporal consistency screening test; low correlation can also be caused by silence, music, dubbing, occlusion, or poor face tracking.",
        "confidence":round(min(90, 35+55*max(best_corr,0)),1),
        "limitations":"This prototype compares a mouth-opening proxy with the audio amplitude envelope. It is not a phoneme-level lip-reading model and cannot prove synthetic speech or a deepfake.",
        "method":"OpenCV mouth proxy + FFmpeg audio RMS envelope cross-correlation",
        "frames_analyzed":frames,
        "mouth_motion_samples":len(mouth_values),
        "audio_duration_seconds":round(float(audio_times[-1]),3),
        "best_correlation":round(float(best_corr),4),
        "best_audio_shift_seconds":round(float(best_lag),3),
    }
