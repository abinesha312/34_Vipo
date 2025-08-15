from pydantic import BaseModel
from pathlib import Path
import os

class Settings(BaseModel):
    # Local data & vector store
    docs_dir: Path = Path("./data")
    chroma_dir: Path = Path("./.chroma")
    chroma_collection: str = "vipo_bank_policies"

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Optional: Google fallback
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    google_llm_model: str = os.getenv("GOOGLE_LLM_MODEL", "gemini-1.5-pro")

settings = Settings()
