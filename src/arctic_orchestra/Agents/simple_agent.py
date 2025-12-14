from typing import Callable, Dict, Any, List
import json


class SimpleAgent:
    """
    Arctic-Orchestra -> Simplified Agent Class (No Tools)

    This version keeps the structure, identity, instruction, task
    but removes:
      - tool schemas
      - tool calling
      - tool validation
      - iterative loop
      - memory or multi-step reasoning

    It simply generates one LLM call using the combined system
    instructions and returns its output directly.
    """

    def __init__(
        self,
        model: Callable[[List[Dict[str, str]]], Dict[str, Any]],
        name: str,
        identity: str,
        instruction: str,
        task: str,
        debug: bool = False,
    ):
        self.model = model
        self.name = name
        self.identity = identity
        self.instruction = instruction
        self.task = task
        self.debug = debug
        self.final_output = ""

    def base_messages(self) -> List[Dict[str, str]]:
        """Create a single system message with identity + instruction + task."""

        system_prompt = (
            f"IDENTITY:\n{self.identity}\n\n"
            f"INSTRUCTION:\n{self.instruction}\n\n"
            f"CURRENT TASK:\n{self.task}\n\n"
            "** Using the instruction solve the task privided by the user **\n"
        )

        return [
            {"role": "system", "content": system_prompt}
        ]

    def _log(self, *args):
        if self.debug:
            print(f"[{self.name}]", *args)

    def run(self, user_input: str) -> str:
        """Send a single message to the model and return output."""

        messages = self.base_messages()
        messages.append({"role": "user", "content": user_input})

        self._log("Sending LLM request:", messages)

        response = self.model(messages)

        # If model returns dict with "content"
        if isinstance(response, dict) and "content" in response:
            output = response["content"]

        # If model returns dict with .text
        elif isinstance(response, dict) and "text" in response:
            output = response["text"]

        # If model returns a raw string
        elif isinstance(response, str):
            output = response

        else:
            output = str(response)

        self.final_output = output
        self._log("LLM reply:", output)

        return output
