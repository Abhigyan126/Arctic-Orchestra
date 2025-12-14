import inspect
import json
import os
import litellm
import warnings
from typing import Callable, Dict, Any, List, Optional, get_type_hints
from arctic_orchestra.Errors.tool_not_supported import ModelToolNotSupportedWarning
from arctic_orchestra.Errors.web_search_not_supported import ModelWebSearchNotSupportedWarning

class Agent:
    """
    A production-grade, universal AI agent wrapper utilizing LiteLLM for cross-provider compatibility.

    This class orchestrates the interaction between a Large Language Model (LLM) and executable Python 
    functions (tools). It manages the conversion of Python type hints to JSON schemas, handles the 
    LLM's tool-calling logic, and maintains a structured context window comprising Identity, 
    Instructions, History, and the current Task.

    Attributes:
        model_name (str): The model identifier string (e.g., 'gpt-4o', 'claude-3-opus', 'gemini-pro').
        name (str): The designation of the agent for logging and identification purposes.
        identity (str): The persona or role definition of the agent (e.g., "You are a Senior DevOps Engineer").
        instruction (str): The set of constraints, guidelines, and behavioral rules the agent must follow.
        tools (Dict[str, Callable]): A dictionary mapping function names to the actual Python callable objects.
        api_key (str): Optional API key override. If None, the environment variable for the specific provider is used.
        debug (bool): Flag to enable verbose logging of the agent's reasoning and execution steps.

    Usage:
        def get_weather(location: str) -> str:
            '''Fetches weather for a location.'''
            return "Sunny, 25C"

        agent = Agent(
            model_name="gpt-4",
            name="WeatherBot",
            identity="You are a helpful assistant.",
            instruction="Always use metric units.",
            tools={"get_weather": get_weather}
        )

        response = agent.run("What is the weather in Paris?")
    """

    def __init__(
        self,
        model_name: str,
        name: str,
        identity: str,
        instruction: str,
        tools: Dict[str, Callable[..., Any]] = None,
        api_key: str = None,
        websearch_config: json = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        debug: bool = False
    ):
        self.model_name = model_name
        self.name = name
        self.identity = identity
        self.instruction = instruction
        self.tools = tools or {}
        self.api_key = api_key
        self.websearch_config = websearch_config
        self.debug = debug
        self.temperature = temperature
        self.top_p = top_p
        self.final_response = ""
        self.tool_schemas = self._build_tool_schemas()

    def _get_type_name(self, t: Any) -> str:
        if t == str: return "string"
        if t == int: return "integer"
        if t == float: return "number"
        if t == bool: return "boolean"
        if t == dict: return "object"
        if t == list: return "array"
        return "string"

    def _debug_log(self, message: str) -> None:
        """Internal helper to print debug messages with agent name prefix."""
        if self.debug:
            print(f"[{self.name}] : {message}")

    def _build_tool_schemas(self) -> List[Dict]:
        schemas = []
        for name, fn in self.tools.items():
            description = inspect.getdoc(fn) or f"Function {name}"
            type_hints = get_type_hints(fn)
            sig = inspect.signature(fn)
            
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name == "self" or param_name.startswith("__"):
                    continue
                
                param_type = type_hints.get(param_name, str)
                json_type = self._get_type_name(param_type)
                
                properties[param_name] = {
                    "type": json_type,
                    "description": f"Parameter {param_name}" 
                }
                
                if param.default == inspect._empty:
                    required.append(param_name)

            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        return schemas

    def _construct_system_prompt(self) -> Dict[str, str]:
        system_content = (
            f"### AGENT IDENTITY\n{self.identity}\n\n"
            f"### OPERATIONAL INSTRUCTIONS\n{self.instruction}"
        )
        return {
            "role": "system",
            "content": system_content
        }

    def run(self, user_input: str, history: List[Dict] = None) -> str:
        if history is None:
            history = []
        
        system_message = self._construct_system_prompt()
        
        formatted_user_task = {
            "role": "user",
            "content": f"### CURRENT TASK\n{user_input}"
        }

        messages = [system_message] + history + [formatted_user_task]

        self._debug_log("--- Initiating Run ---")

        try:
            if (litellm.supports_function_calling(model=self.model_name) == False):
                warnings.warn(ModelToolNotSupportedWarning(
                    error_code='X_2001',
                    data_field=self.model_name,
                    message=f"Model '{self.model_name}' does not support tool calling, use a model with native tool calling support"
                ))
            if (self.websearch_config != None and litellm.supports_web_search(model=self.model_name) == False):
                warnings.warn(ModelWebSearchNotSupportedWarning(
                    error_code='X_2002',
                    data_field=self.model_name,
                ))

            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                tools=self.tool_schemas if self.tool_schemas else None,
                tool_choice="auto" if self.tool_schemas else None,
                api_key=self.api_key,
                temperature=self.temperature,
                top_p=self.top_p,
                web_search_options=self.websearch_config or {}
            )
        except Exception as e:
            return f"LLM Execution Error: {str(e)}"

        response_message = response.choices[0].message
        tool_calls = getattr(response_message, "tool_calls", None)

        # Log assistant's immediate response when in debug mode
        if self.debug:
            try:
                assistant_preview = response_message.content
            except Exception:
                assistant_preview = "<no content>"
            self._debug_log(f"{assistant_preview}")

        if tool_calls:
            messages.append(response_message)

            self._debug_log(f"Tool Execution: Triggered {len(tool_calls)} tools")

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                tool_id = getattr(tool_call, "id", None)

                try:
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except Exception:
                        function_args = {}

                    self._debug_log(
                        f"Tool:'{function_name}' with args: {json.dumps(function_args, ensure_ascii=False)}"
                    )

                    tool_function = self.tools.get(function_name)
                    if not tool_function:
                        raise ValueError(f"Tool {function_name} is not registered.")

                    function_result = tool_function(**function_args)
                    function_response_str = str(function_result)

                    self._debug_log(
                        f"Tool:'{function_name}' returned: {function_response_str}"
                    )

                except Exception as e:
                    function_response_str = f"Error executing tool {function_name}: {str(e)}"
                    self._debug_log(f"Tool:'{function_name}' error: {str(e)}")

                messages.append({
                    "tool_call_id": tool_id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response_str
                })

            final_response = litellm.completion(
                model=self.model_name,
                messages=messages,
                api_key=self.api_key
            )

            if not getattr(final_response, 'choices', None) or not final_response.choices:
                error_msg = f"LLM returned an empty response (no choices) in the final call after tool execution. Model: {self.model_name}"
                self._debug_log(f"!!! ERROR: {error_msg}")
                # You can add logic here to inspect final_response.error if available
                return f"LLM Execution Error: {error_msg}. Please check the logs for potential content filtering or model prediction errors."

            final_content = final_response.choices[0].message.content
            self._debug_log(f" -> {final_content}")
            self.final_response = final_content
            return final_content

        self._debug_log("Returning assistant response without tool calls")
        self.final_response = response_message.content
        return response_message.content