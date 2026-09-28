# DeepTrace — Cyber Cell Investigation Assistant

DeepTrace is a local forensic-analysis prototype for screening suspected manipulated image/video/audio evidence.

## What makes the detector real

The upload workflow runs the **Community Forensics DeepfakeDet-ViT** model locally through ONNX Runtime. It is not a hard-coded/random fake result and it does not depend on a third-party web page for the primary verdict.

Model: https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT

The corrected model uses shortest-edge resize to 440, center-crop to 384, and CLIP normalization. The FP32 ONNX variant is used in this build as the reference model. The model produces a probabilistic synthetic-image score; DeepTrace displays that score and a screening label.

## First run

1. Install Python 3.10+.
2. Run `START_DEEPTRACE.bat` on Windows.
3. The launcher installs the Python requirements.
4. On the first analysis, the backend downloads the ~84 MB FP32 detector model from Hugging Face and caches it under `backend/models/`.
5. Upload an image/video and click **Upload and analyze**.

Internet access is therefore required once to download the model. After it is cached, inference is local.

## Result interpretation

- **LIKELY AI-MANIPULATED / DEEPFAKE SIGNAL**: the model score crossed DeepTrace's screening threshold.
- **NO STRONG AI-GENERATION SIGNAL**: the model did not cross that threshold. This does **not** prove the media is authentic.
- If the model cannot run, DeepTrace explicitly shows **Detector unavailable** instead of inventing a result.

The compression, facial, metadata, and audio/lip modules are supporting evidence. The investigation priority score is a triage score, not a fake probability.

## Independent second opinion

The UI also links to the University at Buffalo DeepFake-O-Meter. It requires its own upload and is separate from DeepTrace's local model result.

## Important limitation

No deepfake detector is guaranteed to identify every manipulation. Model performance can change with new generators, recompression, editing, unusual images, and out-of-distribution media. Use the result as a screening signal and corroborate it with source/provenance and forensic review.
