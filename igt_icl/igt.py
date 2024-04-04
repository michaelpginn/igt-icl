from typing import Optional
from dataclasses import dataclass


@dataclass
class IGT:
    transcription: str
    translation: str
    language: str
    metalang: str
    glosses: Optional[str]
