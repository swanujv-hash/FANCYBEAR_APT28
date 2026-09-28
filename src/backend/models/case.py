from pydantic import BaseModel, Field
from typing import Optional


class CaseCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    officer: str = Field(..., min_length=1, max_length=100)
    evidence_type: str = Field(..., min_length=1, max_length=50)


class CaseResponse(BaseModel):
    case_id: str
    title: str
    officer: str
    evidence_type: str
    status: str
    created_at: str


class EvidenceResponse(BaseModel):
    evidence_id: int
    case_id: str
    filename: str
    file_hash: str
    file_size: int
    uploaded_at: str


class AnalysisResponse(BaseModel):
    case_id: str
    evidence_id: int
    priority_score: Optional[float]
    metadata: dict
    compression: dict
    facial_landmarks: dict
    audio_lip_sync: dict
    created_at: str
