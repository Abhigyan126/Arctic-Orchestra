import os
import json
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.auth import load_credentials_from_file, default as default_credentials


class GeminiClient:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        config_args: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[List[types.SafetySetting]] = None,
        vertex_ai: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
        service_account_json: Optional[str] = None,
        load_env: bool = True,
        grounding: bool = False
    ):
        if load_env:
            load_dotenv()

        self.model = model
        self.temperature = temperature
        self.config_args = config_args or {}
        self.safety_settings = safety_settings
        self.vertex_ai = vertex_ai
        self.grounding = grounding

        if vertex_ai:
            self._init_vertex_ai(project, location, service_account_json)
        else:
            self._init_public_api(api_key)

    def _init_public_api(self, api_key: Optional[str]):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing Gemini API key (GEMINI_API_KEY).")

        self.client = genai.Client(api_key=self.api_key)
        self.project = None
        self.location = None

    def _init_vertex_ai(
        self,
        project: Optional[str],
        location: Optional[str],
        service_account_json: Optional[str],
    ):
        if not project or not location:
            raise ValueError("Vertex AI requires BOTH project and location.")

        self.project = project
        self.location = location
        vertex_scopes = ["https://www.googleapis.com/auth/cloud-platform"]

        if service_account_json:
            if not os.path.exists(service_account_json):
                raise FileNotFoundError(f"Service account JSON not found: {service_account_json}")

            credentials, _ = load_credentials_from_file(service_account_json, scopes=vertex_scopes)
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                credentials=credentials
            )
            return

        try:
            credentials, _ = default_credentials(scopes=vertex_scopes)
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                credentials=credentials
            )
        except Exception:
            raise ValueError(
                "Vertex AI authentication failed.\n"
                "Provide service_account_json OR run:\n"
                "    gcloud auth application-default login\n"
            )

    def _clean_text(self, text: str) -> str:
        """Strips Markdown code blocks (```json ... ```) from the response text."""
        cleaned = text.strip()
        # Check if text is wrapped in code fences
        if cleaned.startswith("```") and cleaned.endswith("```"):
            # Split by the first newline to separate the opening fence (e.g., ```json)
            parts = cleaned.split("\n", 1)
            if len(parts) > 1:
                # Take the content part
                content = parts[1]
                # Remove the trailing ``` (last 3 chars)
                content = content[:-3]
                return content.strip()
        return cleaned

    def run(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        config_args: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[List[types.SafetySetting]] = None,
        grounding: Optional[bool] = None,
    ) -> Union[Dict[str, Any], str]:

        system_instruction = None
        api_messages = []

        for m in messages:
            role = m.get("role", "user").lower()
            content = m.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "assistant":
                api_messages.append(types.Content(role="model", parts=[types.Part(text=content)]))
            elif role == "model":
                api_messages.append(types.Content(role="model", parts=[types.Part(text=content)]))
            else:
                api_messages.append(types.Content(role="user", parts=[types.Part(text=content)]))

        final_config = {**self.config_args, **(config_args or {})}

        # --- Grounding Logic ---
        should_ground = grounding if grounding is not None else self.grounding

        if should_ground:
            search_tool = types.Tool(google_search=types.GoogleSearch())
            current_tools = final_config.get("tools")
            if current_tools is None:
                final_config["tools"] = [search_tool]
            elif isinstance(current_tools, list):
                final_config["tools"] = current_tools + [search_tool]

        if system_instruction:
            final_config["system_instruction"] = system_instruction

        final_temperature = temperature if temperature is not None else self.temperature
        final_safety = safety_settings or self.safety_settings

        generation_config = types.GenerateContentConfig(
            temperature=final_temperature,
            safety_settings=final_safety,
            **final_config
        )

        try:
            response = self.client.models.generate_content(
                model=model or self.model,
                contents=api_messages,
                config=generation_config,
            )

            # Get text and clean potential Markdown fences
            text = self._clean_text(response.text)

            if final_config.get("response_mime_type") == "application/json":
                try:
                    return json.loads(text)
                except Exception:
                    # If parsing fails, return the cleaned text
                    return text

            return text

        except Exception as e:
            return {"error": str(e)}

    def configure(
        self,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        config_args: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[List[types.SafetySetting]] = None,
        grounding: Optional[bool] = None
    ):
        if model:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if config_args is not None:
            self.config_args.update(config_args)
        if safety_settings is not None:
            self.safety_settings = safety_settings
        if grounding is not None:
            self.grounding = grounding