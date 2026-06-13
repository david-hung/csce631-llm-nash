"""
TAMU API client for querying LLMs.
Set TAMU_API_COOKIE and TAMU_API_BASE_URL in .env before use.
"""
import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()


class TAMUClient:
    def __init__(self, model: str = "gpt-4o", temperature: float = 1.0):
        self.base_url = os.getenv("TAMU_API_BASE_URL", "https://llm.tamu.edu")
        self.cookie = os.getenv("TAMU_API_COOKIE", "")
        self.model = model
        self.temperature = temperature
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Cookie": self.cookie,
        })

    def query(self, prompt: str, system_prompt: str = "", retries: int = 3) -> str:
        """Send a prompt and return the model's text response."""
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [],
        }
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({"role": "user", "content": prompt})

        for attempt in range(retries):
            try:
                resp = self.session.post(
                    f"{self.base_url}/api/chat/completions",
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"TAMU API error after {retries} attempts: {e}")
        return ""

    def query_game(self, game_prompt: str, temperature: float | None = None) -> str:
        """Query a game scenario with an optional temperature override."""
        if temperature is not None:
            original = self.temperature
            self.temperature = temperature
            result = self.query(game_prompt, system_prompt=SYSTEM_PROMPT)
            self.temperature = original
            return result
        return self.query(game_prompt, system_prompt=SYSTEM_PROMPT)


SYSTEM_PROMPT = """You are a rational decision-maker participating in a game theory experiment.
When presented with a game scenario, you will be asked to choose an action or specify a mixed strategy.
Always respond with ONLY the action name or a JSON probability distribution over actions.
Do not add explanations unless explicitly asked."""
