import os
import json
import requests
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv


class OpenRouterClientQwen:
    """
    A flexible OpenRouter API client.

    - Loads API key from environment (OPENROUTER_API_KEY) OR accepts custom key on init.
    - Configurable model, temperature, response format, timeout.
    - Allows reuse across agents and models.

    ### usage

    .run(message)
    .configure(
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        )
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        load_env: bool = True,
    ):
        if load_env:
            load_dotenv()  # Load .env automatically

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided and not found in .env")

        self.model = model
        self.temperature = temperature
        self.response_format = response_format or {"type": "json_object"}
        self.timeout = timeout
        self.base_url = base_url

    def run(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], str]:
        """
        Send chat completion request to OpenRouter.
        Allows overriding model, temperature, etc.
        """

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        # Add response format only if provided
        if response_format or self.response_format:
            payload["response_format"] = response_format or self.response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.base_url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Try to parse JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content  # raw string, agent can handle

        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}

        except (KeyError, IndexError):
            return {"error": "Unexpected API response format"}

    def configure(
        self,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ):
        """Dynamically update client settings."""
        if model:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if response_format is not None:
            self.response_format = response_format
        if timeout is not None:
            self.timeout = timeout
