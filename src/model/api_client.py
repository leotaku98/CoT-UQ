# -*- coding: utf-8 -*-
"""OpenAI-compatible API client for featherless.ai and OpenAI providers."""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def chat_complete(
    prompt: str,
    model_id: str,
    provider: str,
    temperature: float = 1.0,
    max_new_tokens: int = 256,
    top_p: float = 0.9,
    **_kwargs,
) -> str:
    """Send a single-turn chat prompt and return the response text.

    Raises:
        ValueError: If provider is not recognized.
        Exception: Re-raised after 3 failed retries with exponential backoff.
    """
    if provider == "featherless":
        api_key = os.environ["FEATHERLESS_API_KEY"]
        base_url = os.environ["FEATHERLESS_BASE_URL"]
    elif provider == "openai":
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = os.environ["OPENAI_BASE_URL"]
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Expected 'featherless' or 'openai'.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=top_p,
            )
            return response.choices[0].message.content
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
