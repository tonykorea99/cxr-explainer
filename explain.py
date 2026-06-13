import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from config import GEMINI_MODEL, SYSTEM_INSTRUCTION

load_dotenv()

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and fill in the key.")
        _client = genai.Client(api_key=api_key)
    return _client


def explain(findings: str, question: str | None = None) -> str:
    client = _get_client()

    user_content = f"[흉부 X-ray 소견]\n{findings}"
    if question:
        user_content += f"\n\n[추가 질문]\n{question}"

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
        ),
        contents=user_content,
    )
    return response.text
