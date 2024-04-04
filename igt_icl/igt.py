from typing import Optional
from dataclasses import dataclass


@dataclass
class IGT:
    transcription: str
    translation: Optional[str]
    language: str
    metalang: Optional[str]
    glosses: Optional[str]

    def __str__(self) -> str:
        s = f"Transcription: {self.transcription}"
        if self.translation is not None:
            s = s + f"\nTranslation: {self.translation}"
        if self.glosses is not None:
            s = s + f"\Glosses: {self.glosses}"
        return s
