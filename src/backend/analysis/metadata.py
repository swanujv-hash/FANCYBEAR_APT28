from pathlib import Path
import json
import subprocess


def _run_ffprobe(path: Path):
    command = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height,r_frame_rate",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def extract_metadata(path: Path) -> dict:
    ffprobe_data = _run_ffprobe(path)

    result = {
        "status": "review",
        "finding": "Basic file metadata extracted.",
        "observation": "",
        "significance": "Metadata describes the supplied file but does not establish authenticity.",
        "confidence": None,
        "limitations": "Metadata can be changed or removed during editing, transcoding, upload, or export.",
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
    }

    if not ffprobe_data:
        result["observation"] = "FFmpeg/ffprobe metadata was not available; basic file information was collected."
        result["limitations"] += " Install FFmpeg and ensure ffprobe is on PATH for detailed media metadata."
        return result

    fmt = ffprobe_data.get("format", {})
    streams = ffprobe_data.get("streams", [])

    result["duration_seconds"] = float(fmt["duration"]) if fmt.get("duration") else None
    result["format"] = fmt.get("format_name")

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video:
        result["video"] = {
            "codec": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "frame_rate": video.get("r_frame_rate"),
        }
    else:
        result["video"] = None

    result["audio"] = {
        "codec": audio.get("codec_name")
    } if audio else None

    result["observation"] = "Container and stream metadata extracted successfully."
    return result
