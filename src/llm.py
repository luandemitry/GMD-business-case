import os
from typing import Protocol

from dotenv import load_dotenv

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLM(Protocol):
    name: str

    def generate(self, system: str, prompt: str, json_output: bool = False) -> str: ...


def get_secret(name: str, default: str | None = None) -> str | None:
    load_dotenv()
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


class GeminiLLM:
    def __init__(self, api_key: str, model: str):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.name = f"gemini:{model}"

    def generate(self, system: str, prompt: str, json_output: bool = False) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json" if json_output else "text/plain",
        )
        response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        return response.text or ""


class GroqLLM:
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        self.model = model
        self.name = f"groq:{model}"

    def generate(self, system: str, prompt: str, json_output: bool = False) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if json_output else None,
        )
        return response.choices[0].message.content or ""


class FallbackLLM:
    def __init__(self, providers: list[LLM]):
        self.providers = providers
        self.name = " -> ".join(p.name for p in providers)
        self.last_used: str | None = None

    def generate(self, system: str, prompt: str, json_output: bool = False) -> str:
        errors = []
        for provider in self.providers:
            try:
                result = provider.generate(system, prompt, json_output)
                self.last_used = provider.name
                return result
            except Exception as exc:
                errors.append(f"{provider.name}: {type(exc).__name__}")
        raise RuntimeError("All LLM providers failed: " + "; ".join(errors))


def build_llm() -> FallbackLLM | None:
    providers: list[LLM] = []
    if key := get_secret("GEMINI_API_KEY"):
        providers.append(GeminiLLM(key, get_secret("GEMINI_MODEL", "gemini-3.6-flash")))
    if key := get_secret("GROQ_API_KEY"):
        providers.append(GroqLLM(key, get_secret("GROQ_MODEL", "openai/gpt-oss-120b")))
    return FallbackLLM(providers) if providers else None
