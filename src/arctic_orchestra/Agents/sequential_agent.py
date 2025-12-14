import json
from typing import List, Dict, Any


def compress_with_model(model, entries: List[dict]) -> str:
    """Model-based summarization of memory entries."""
    prompt = (
        "Summarize the following agent outputs into a concise paragraph "
        "(max 150 words). Keep key decisions and results.\n\n"
        f"{json.dumps(entries, indent=2)}"
    )
    try:
        resp = model([{"role": "user", "content": prompt}])
        return resp if isinstance(resp, str) else resp.get("content", "...")
    except Exception:
        return " | ".join(f"{e['agent']}: {e['output'][:200]}" for e in entries)


class SequentialAgent:
    """
    New SequentialAgent designed to work with the updated Agent class.
    - No JSON contract
    - No additional_instruction passing
    - Only memory accumulation + compression
    """

    def __init__(
        self,
        name: str,
        description: str,
        agents: List[Any],        # List of Agent instances
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

    # ---------------------------------------------------------------------
    # Memory Handling
    # ---------------------------------------------------------------------

    def _compress_memory(self, entries: List[Dict[str, str]]) -> str:
        """Compress old memory entries using model if available."""
        if not entries:
            return ""

        if self.compression_model:
            return compress_with_model(self.compression_model, entries)

        # fallback
        return " | ".join(
            f"{e['agent']}: {e['output'][:200]}" for e in entries
        )

    def _add_memory(self, agent_name: str, output: str):
        entry = {"agent": agent_name, "output": output}

        self.long_memory.append(entry)
        self.short_memory.append(entry)

        self._enforce_memory_limits()

    def _enforce_memory_limits(self):
        """If short-memory exceeds threshold: compress older entries."""
        packed = json.dumps(self.short_memory)

        # If still small enough → no need to compress
        if len(packed) < self.max_context_chars:
            return

        # Compress everything except last `window_size` messages
        if len(self.short_memory) > self.window_size:
            old = self.short_memory[:-self.window_size]
            new = self.short_memory[-self.window_size:]

            compressed_text = self._compress_memory(old)

            # Replace old content with compressed chunk
            self.short_memory = [{"compressed": compressed_text}] + new

        # Safety: if still too large → keep only final window_size
        packed = json.dumps(self.short_memory)
        if len(packed) > self.max_context_chars:
            self.short_memory = self.short_memory[-self.window_size:]

    # ---------------------------------------------------------------------
    # Main Execution Flow
    # ---------------------------------------------------------------------

    def run(self, user_query: str) -> str:
        """
        Runs agents sequentially.
        Each agent receives:
            - Original user query
            - Short memory (compressed if needed)
            - Long memory
        """
        # Reset run-specific memories
        self.long_memory = []
        self.short_memory = []

        for i, agent in enumerate(self.agents, start=1):

            # Construct context for this agent
            agent_input = {
                "original_query": user_query,
                "step_number": i,
                "agent_name": agent.name,
                "short_memory": self.short_memory,
                "long_memory": self.long_memory,
            }

            # Run the agent
            response = agent.run(json.dumps(agent_input))

            # Update memory
            self._add_memory(agent.name, response)

        # The output of the final agent is final result
        return self.long_memory[-1]["output"]
