"""
TAMU API client using the OpenAI-compatible gateway at chat.tamu.ai.

Setup:
  1. Log into https://chat.tamu.ai with your NetID + Duo MFA
  2. Extract CF_Authorization cookie from DevTools (Application tab → Cookies)
  3. Add to .env:  TAMU_CF_COOKIE=CF_Authorization=eyJ...

Claude models (Sonnet, Opus) require temperature=1 and max_tokens>=16384 due to
extended thinking mode. Use GPT or Haiku models for temperature sweep experiments.
"""
import os
import time
import openai
from dotenv import load_dotenv

load_dotenv()

TAMU_BASE_URL = "https://chat.tamu.ai/api"

# Models available on the TAMU gateway
MODELS = {
    "sonnet": "protected.Claude Sonnet 4.5",   # best quality; temp must be 1, max_tokens>=16384
    "haiku": "protected.Claude-Haiku-4.5",     # fast Claude; any temp, any max_tokens
    "opus": "protected.Claude Opus 4.5",        # strongest; temp must be 1, max_tokens>=16384
    "gpt-mini": "protected.gpt-5-mini",         # cheap GPT; any temp, max_tokens<=256
    "gpt": "protected.gpt-5.2",                 # strong GPT; any temp
    "gemini-flash": "protected.gemini-2.5-flash",
    "gemini-pro": "protected.gemini-2.5-pro",
}

# Models that require thinking mode constraints
THINKING_MODELS = {"protected.Claude Sonnet 4.5", "protected.Claude Opus 4.5"}


def _make_client() -> openai.OpenAI:
    cf_cookie = os.getenv("TAMU_CF_COOKIE", "")
    api_key = os.getenv("TAMU_API_KEY", "")
    if not cf_cookie:
        raise EnvironmentError(
            "TAMU_CF_COOKIE not set in .env. "
            "Log into chat.tamu.ai, copy the CF_Authorization cookie, "
            "then add TAMU_CF_COOKIE=CF_Authorization=eyJ... to your .env file."
        )
    if not api_key:
        raise EnvironmentError("TAMU_API_KEY not set in .env.")
    return openai.OpenAI(
        base_url=TAMU_BASE_URL,
        api_key=api_key,
        default_headers={"Cookie": cf_cookie},
    )


class TAMUClient:
    def __init__(self, model: str = "gpt-mini", temperature: float = 0.7):
        """
        model: key from MODELS dict (e.g. 'gpt-mini', 'haiku', 'sonnet')
               or a full model identifier string
        temperature: ignored for thinking-mode Claude models (forced to 1)
        """
        self.model_id = MODELS.get(model, model)
        self.temperature = temperature
        self.client = _make_client()
        self._total_tokens = 0

    def query(self, prompt: str, system_prompt: str = "", retries: int = 3) -> str:
        is_thinking = self.model_id in THINKING_MODELS
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = dict(
            model=self.model_id,
            messages=messages,
            temperature=1 if is_thinking else self.temperature,
            max_tokens=16384 if is_thinking else 512,
        )

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._total_tokens += getattr(response.usage, "total_tokens", 0)
                return response.choices[0].message.content or ""
            except openai.RateLimitError:
                time.sleep(2 ** attempt)
            except openai.AuthenticationError:
                raise RuntimeError(
                    "401/403: Cookie expired. Log into chat.tamu.ai and refresh TAMU_CF_COOKIE in .env."
                )
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                else:
                    raise RuntimeError(f"TAMU API error after {retries} attempts: {e}")
        return ""

    def query_game(self, game_prompt: str, temperature: float | None = None) -> str:
        original = self.temperature
        if temperature is not None:
            self.temperature = temperature
        result = self.query(game_prompt, system_prompt=GAME_SYSTEM_PROMPT)
        self.temperature = original
        return result

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens


GAME_SYSTEM_PROMPT = (
    "You are a rational decision-maker in a game theory experiment. "
    "When shown a payoff matrix, choose the action that maximizes your expected payoff. "
    "Respond with ONLY the action name. "
    "If you want to play a mixed strategy, respond with a JSON object like "
    '{\"Action1\": 0.6, \"Action2\": 0.4}. No explanations.'
)
