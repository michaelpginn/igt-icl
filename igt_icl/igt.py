from typing import Optional, Dict
from dataclasses import dataclass, fields


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

    @classmethod
    def from_dict(cls, dict: Dict):
        dataclass_fields = {field.name for field in fields(IGT)}
        filtered_dict = {key: value for key, value in dict.items() if key in dataclass_fields}
        return IGT(**filtered_dict)
