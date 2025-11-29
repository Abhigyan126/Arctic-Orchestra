import json
from typing import List, Dict, Any


def compress_with_model(model, entries: List[dict]) -> str:
    """Your improved compression helper."""
    prompt = (
        "Summarize the following list of agent outputs into a concise "
        "paragraph (max 150 words). Preserve the key decisions and results.\n\n"
        f"{json.dumps(entries, indent=2)}"
    )
    try:
        resp = model([{"role": "user", "content": prompt}])
        return resp if isinstance(resp, str) else resp.get("content", "...")
    except Exception:
        return " | ".join(f"{e['agent']}: {e['output'][:200]}" for e in entries)


class SequentialAgent:
    def __init__(
        self,
        name: str,
        description: str,
        agents: List[Any],            
        compression_model=None,
        window_size: int = 2,
        max_context_chars: int = 8000,
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.compression_model = compression_model
        self.window_size = window_size
        self.max_context_chars = max_context_chars

        self.long_memory = []
        self.short_memory = []

    def _compress_memory(self, entries: List[Dict[str, str]]) -> str:
        if not entries:
            return ""

        # use provided model if available
        if self.compression_model:
            return compress_with_model(self.compression_model, entries)

        # fallback: cheap truncation
        return " | ".join(
            f"{e['agent']}: {e['output'][:200]}" for e in entries
        )

    def _add_memory(self, agent_name: str, output: str):
        entry = {"agent": agent_name, "output": output}

        self.long_memory.append(entry)
        self.short_memory.append(entry)

        self._enforce_memory_limits()

    def _enforce_memory_limits(self):
        packed = json.dumps(self.short_memory)

        if len(packed) < self.max_context_chars:
            return

        # compress
        if len(self.short_memory) > self.window_size:
            old = self.short_memory[:-self.window_size]
            new = self.short_memory[-self.window_size:]
            compressed = self._compress_memory(old)
            self.short_memory = [{"compressed": compressed}] + new

        # emergency trim
        packed = json.dumps(self.short_memory)
        if len(packed) > self.max_context_chars:
            self.short_memory = self.short_memory[-self.window_size:]

    def run(self, user_query: str) -> str:
        self.long_memory = []
        self.short_memory = []

        forwarded_instruction = None

        for i, agent in enumerate(self.agents, start=1):

            contract = (
                "When producing your final answer, YOU MUST output strict JSON:\n"
                "{\n"
                '   "final_output": "<your main generated content>",\n'
                '   "additional_instruction": "<optional guidance for next agent or empty string>"\n'
                "}\n\n"
                "Rules:\n"
                "- Do NOT include explanation outside JSON.\n"
                "- `additional_instruction` may be empty if not needed.\n"
                "- Follow the context provided.\n"
                "- Incorporate forwarded instruction if given.\n"
            )

            step_input = {
                "original_query": user_query,
                "step_number": i,
                "agent_name": agent.name,
                "previous_context": self.short_memory,
                "additional_instruction_from_previous_agent": forwarded_instruction,
                "contract": contract,
            }

            # Run agent (BaseAgent expects a string)
            response = agent.run(json.dumps(step_input))

            # Update memory
            self._add_memory(agent.name, response)

            # Try extracting forwarded instruction
            try:
                parsed = json.loads(response)
                forwarded_instruction = parsed.get("additional_instruction", None)
            except Exception:
                forwarded_instruction = None

        return self.long_memory[-1]["output"]
