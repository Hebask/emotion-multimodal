from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    TEXT_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"
    AUDIO_MODEL: str = "superb/wav2vec2-base-superb-er"

    DEVICE: str = "cpu"

settings = Settings()
