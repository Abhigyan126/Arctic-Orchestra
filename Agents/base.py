from typing import Callable, Dict, Any, List, Optional, Union
import json
import inspect


class Agent:
    """
    Arctic-Orchestra -> Agent Class

    A versatile Base Agent designed to orchestrate LLM-based tool execution for complex tasks.
    It provides the LLM with a clear identity, strict operational instructions, and a specific task.
    These system-level directives guide the agent's behavior, while user-level interactions
    are handled via the `run` method.

    The agent operates iteratively: it sends the conversation history to the LLM, receives
    JSON-formatted responses (either tool calls or final answers), validates them, executes
    tools if necessary, and feeds the results back to the LLM.

    ## Methods

    run(user_input: str) -> str
    Starts the iterative execution loop to process the user's request
    and return the final output.

    ## Note 
    You can use the final_output variable to fetch the final output from the agent.
    It also does return the same, this may come handy when usign it with other available tools.    
    """

    def __init__(
        self,
        model: Callable[[List[Dict[str, str]]], Dict[str, Any]],
        name: str,
        identity: str,
        instruction: str,
        task: str,
        tools: Dict[str, Callable[..., Any]] = None,
        max_iterations: int = 10,
        debug: bool = False,
        final_output: str = "",
    ):
        self.model = model
        self.name = name
        self.identity = identity
        self.instruction = instruction
        self.task = task
        self.tools = tools or {}
        self.max_iterations = max_iterations
        self.debug = debug
        self.final_output = final_output
        self.tool_schemas = self._build_tool_schemas()

    def _build_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Build detailed schemas for each tool including parameter types."""
        schemas = {}
        for name, fn in self.tools.items():
            try:
                # Get tool description
                desc, usage = fn("__describe__")
                
                # Extract parameter information from function signature
                sig = inspect.signature(fn)
                params = {}
                for param_name, param in sig.parameters.items():
                    if param_name == "kwargs" or param_name.startswith("__"):
                        continue
                    params[param_name] = {
                        "type": param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any",
                        "required": param.default == inspect.Parameter.empty
                    }
                
                schemas[name] = {
                    "description": desc,
                    "example": usage,
                    "parameters": params
                }
            except Exception as e:
                self._log(f"Warning: Could not build schema for tool '{name}': {e}")
                schemas[name] = {
                    "description": "No description available",
                    "example": "No example available",
                    "parameters": {}
                }
        
        return schemas

    def _format_tool_schemas(self) -> str:
        """Format tool schemas in a clear, structured way."""
        if not self.tool_schemas:
            return "No tools available."
        
        formatted = []
        for name, schema in self.tool_schemas.items():
            tool_info = [f"Tool: {name}"]
            tool_info.append(f"Description: {schema['description']}")
            
            if schema['parameters']:
                tool_info.append("Parameters:")
                for param_name, param_info in schema['parameters'].items():
                    required = "required" if param_info['required'] else "optional"
                    tool_info.append(f"  - {param_name} ({param_info['type']}, {required})")
            else:
                tool_info.append("Parameters: None")
            
            tool_info.append(f"Example: {schema['example']}")
            formatted.append("\n".join(tool_info))
        
        return "\n\n".join(formatted)

    def base_messages(self) -> List[Dict[str, str]]:
        """Build base system messages with clear instructions and examples."""
        tool_schemas_str = self._format_tool_schemas()
        
        return [
            {
                "role": "system",
                "content": (
                    f"IDENTITY:\n{self.identity}\n\n"
                    f"INSTRUCTION:\n{self.instruction}\n\n"
                    f"CURRENT TASK:\n{self.task}\n\n"
                    "=== CRITICAL RULES ===\n"
                    "1. You MUST respond with valid JSON only - no other text\n"
                    "2. You MUST use only the tools listed below - never invent tool names\n"
                    "3. You MUST match parameter names exactly as specified\n"
                    "4. Always call appropriate tools before providing final answers\n"
                    "5. Verify tool outputs before finishing\n\n"
                    "=== RESPONSE PROTOCOLS ===\n"
                    "To call a tool:\n"
                    '{"tool": "<exact_tool_name>", "args": {"param1": "value1", "param2": "value2"}}\n\n'
                    "To finish the task:\n"
                    '{"finish": true, "output": "your final answer here"}\n\n'
                    "=== AVAILABLE TOOLS ===\n"
                    f"{tool_schemas_str}\n\n"
                    "=== EXAMPLES ===\n"
                    "Good response (tool call):\n"
                    '{"tool": "calculator", "args": {"expression": "2 + 2"}}\n\n'
                    "Good response (finish):\n"
                    '{"finish": true, "output": "The calculation result is 4"}\n\n'
                    "BAD responses (DO NOT DO):\n"
                    '- {"tool": "made_up_tool", "args": {}} ← Tool does not exist\n'
                    '- {"tool": "calculator", "args": {"expr": "2+2"}} ← Wrong parameter name\n'
                    '- Let me calculate that... {"tool": "calculator"} ← Extra text before JSON'
                ),
            }
        ]

    def _validate_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """Validate a tool call against the schema. Returns error message or None."""
        if tool_name not in self.tool_schemas:
            available = ", ".join(self.tool_schemas.keys())
            return f"Tool '{tool_name}' does not exist. Available tools: {available}"
        
        schema = self.tool_schemas[tool_name]
        
        # Check required parameters
        for param_name, param_info in schema['parameters'].items():
            if param_info['required'] and param_name not in args:
                return f"Missing required parameter '{param_name}' for tool '{tool_name}'"
        
        # Check for unexpected parameters
        expected_params = set(schema['parameters'].keys())
        provided_params = set(args.keys())
        unexpected = provided_params - expected_params
        
        if unexpected:
            return f"Unexpected parameters for tool '{tool_name}': {', '.join(unexpected)}. Expected: {', '.join(expected_params)}"
        
        return None

    def _parse_response(self, response: Union[Dict[str, Any], str]) -> Union[Dict[str, Any], str]:
        """Parse and validate model response."""
        # If already a dict, return as-is
        if isinstance(response, dict):
            return response
        
        # If string, try to parse as JSON
        if isinstance(response, str):
            response = response.strip()
            
            # Try to extract JSON if there's extra text
            if not response.startswith('{'):
                # Look for JSON object in the response
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1:
                    response = response[start:end+1]
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"error": "Could not parse response as JSON", "raw": response}
        
        return response

    def _log(self, *args):
        """Debug logging."""
        if self.debug:
            print(f"[{self.name}]", *args)

    def run(self, user_input: str) -> str:
        messages = self.base_messages()
        messages.append({"role": "user", "content": user_input})

        for step in range(self.max_iterations):
            print("\n")
            self._log(f"[{step}]")
            raw_response = self.model(messages)
            self._log(f"Raw model response: {raw_response}")

            response = self._parse_response(raw_response)
            self._log(f"Parsed response: {json.dumps(response, indent=2, ensure_ascii=False)}")

            match response:
                case {"error": err, **rest}:
                    msg = (
                        f"Failed to parse your response as JSON. "
                        f"You must respond with valid JSON only.\n"
                        f"Error: {err}\n"
                        f"Your response was: {rest.get('raw', 'N/A')}"
                    )
                    messages.append({"role": "assistant", "content": str(raw_response)})
                    messages.append({"role": "user", "content": msg})

                case {"finish": True, "output": output}:
                    self._log("Task finished.")
                    self.final_output = output
                    return str(output)

                case {"tool": tool_name, "args": args}:
                    validation_error = self._validate_tool_call(tool_name, args)
                    if validation_error:
                        self._log(f"Validation error: {validation_error}")
                        messages.append({"role": "assistant", "content": json.dumps(response)})
                        messages.append({"role": "user", "content": validation_error})

                    try:
                        self._log(f"Executing: {tool_name}({args})")
                        tool_result = self.tools[tool_name](**args)
                        self._log(f"Tool result: {tool_result}")
                        messages.append({"role": "assistant", "content": json.dumps(response)})
                        messages.append({"role": "tool", "content": str(tool_result)})
                    except Exception as e:
                        m = f"Error executing tool '{tool_name}': {str(e)}"
                        self._log(m)
                        messages.append({"role": "assistant", "content": json.dumps(response)})
                        messages.append({"role": "tool", "content": m})

                case s if isinstance(s, str):
                    self._log("Received direct text response (not JSON).")
                    return s

                case _:
                    m = (
                        f"Invalid response format. Your response must be one of:\n"
                        f'1. Tool call: {{"tool": "name", "args": {{}}}}\n'
                        f'2. Finish: {{"finish": true, "output": "result"}}\n'
                        f"Your response: {response}"
                    )
                    messages.append({"role": "assistant", "content": str(raw_response)})
                    messages.append({"role": "user", "content": m})

        return f"Error: Task not completed within {self.max_iterations} iterations."
