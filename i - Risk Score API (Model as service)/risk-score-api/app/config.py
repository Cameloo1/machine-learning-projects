"""Configuration settings for the Alert Risk Score API."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Base directory (project root)
    BASE_DIR: Path = Path(__file__).parent.parent
    
    # Model paths
    MODEL_PATH: Path = BASE_DIR / "models" / "model.joblib"
    MODEL_META_PATH: Path = BASE_DIR / "models" / "model_meta.json"
    METRICS_PATH: Path = BASE_DIR / "metrics" / "metrics.json"
    
    # Environment
    ENV: str = "local"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()

