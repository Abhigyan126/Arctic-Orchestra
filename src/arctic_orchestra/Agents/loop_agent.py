import json
import re
from typing import List, Dict, Any
from .sequential_agent import SequentialAgent


class LoopSequentialAgent(SequentialAgent):
    def __init__(
        self,
        name: str,
        description: str,
        agents: List[Any],
        compression_model=None,
        window_size: int = 2,
        max_context_chars: int = 8000,
        max_loops: int = 10,
        local_memory_window: int = 5,
        agents_with_exit_flag: List[Any] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            agents=agents,
            compression_model=compression_model,
            window_size=window_size,
            max_context_chars=max_context_chars,
        )
        self.max_loops = max_loops
        self.local_memory_window = local_memory_window

        # Persistent memory per agent across loops
        self.agent_persistent_memory = {agent.name: [] for agent in agents}

        # Loop control
        self.loop_flag = True

        # Agents allowed to end loop
        self.agents_with_exit_flag = agents_with_exit_flag or []

    # ----------------------------------------
    # Memory Utils
    # ----------------------------------------
    def _add_to_agent_memory(self, agent_name: str, loop_cycle: int, output: str):
        entry = {"loop_cycle": loop_cycle, "output": output}
        self.agent_persistent_memory[agent_name].append(entry)

        if len(self.agent_persistent_memory[agent_name]) > self.local_memory_window:
            self.agent_persistent_memory[agent_name].pop(0)

    # ----------------------------------------
    # Contract Builder
    # ----------------------------------------
    def _build_contract(self, agent, has_exit_privilege: bool) -> str:
        base_contract = (
            "When producing your final answer, YOU MUST output strict JSON:\n"
            "Output must be placed inside the {'finish': true, 'output': '...'} wrapper.\n"
            "The value of `output` MUST be a JSON object with the following structure:\n"
            "** The Json below must be strictly inside the ouput object in the wrapper **"
            "{\n"
            '   "final_output": "<your main generated content>",\n'
            '   "additional_instruction": "<optional guidance or empty string>"'
        )

        # Only privileged agents get the termination field
        if has_exit_privilege:
            base_contract += ',\n   "terminate_loop": <true or false>\n'
        else:
            base_contract += "\n"

        base_contract += "}\n\nRules:\n"
        base_contract += "- ONLY output the JSON object. No text outside JSON.\n"
        base_contract += "- Do NOT escape the inner JSON.\n"
        base_contract += "- `additional_instruction` must be text.\n"

        # Only privileged agents get termination capabilities
        if has_exit_privilege:
            base_contract += "- `terminate_loop` controls loop ending.\n"
            base_contract += "- You may also embed `<terminate>true</terminate>` inside `final_output` whenever you do terminate: true.\n"

        return base_contract

    # ----------------------------------------
    # TERMINATION DETECTION (ROBUST)
    # ----------------------------------------
    def _detect_xml_termination(self, response: str) -> bool:
        """
        Detects <terminate>true</terminate> anywhere in output.
        Case-insensitive and whitespace-tolerant.
        """
        return bool(re.search(r"<\s*terminate\s*>\s*true\s*<\s*/\s*terminate\s*>", response, re.IGNORECASE))

    def _detect_json_termination(self, response: str) -> bool:
        """
        Extract `terminate_loop` from JSON safely.
        JSON may be broken — fallback to False.
        """
        try:
            parsed = json.loads(response)
            return parsed.get("terminate_loop", False)
        except Exception:
            return False

    def check_termination(self, response: str, has_exit_privilege: bool) -> bool:
        """
        CENTRAL termination logic combining both JSON and XML detection.
        Only applies if agent has exit privilege.
        """
        if not has_exit_privilege:
            return False

        # Priority 1: JSON termination flag (cleanest)
        if self._detect_json_termination(response):
            return True

        # Priority 2: XML termination tag (backup)
        if self._detect_xml_termination(response):
            return True

        return False

    # ----------------------------------------
    # MAIN LOOP EXECUTION
    # ----------------------------------------
    def run(self, user_query: str) -> str:
        # Reset memory
        self.long_memory = []
        self.short_memory = []
        self.agent_persistent_memory = {agent.name: [] for agent in self.agents}
        self.loop_flag = True

        forwarded_instruction = None
        loop_cycle = 0

        while self.loop_flag and loop_cycle < self.max_loops:
            loop_cycle += 1

            for step_index, agent in enumerate(self.agents, start=1):

                if not self.loop_flag:
                    break

                has_exit_privilege = agent in self.agents_with_exit_flag
                contract = self._build_contract(agent, has_exit_privilege)

                step_input = {
                    "original_query": user_query,
                    "loop_cycle": loop_cycle,
                    "step_number": step_index,
                    "agent_name": agent.name,
                    "agent_local_memory": self.agent_persistent_memory[agent.name],
                    "global_short_memory": self.short_memory,
                    "additional_instruction_from_previous_agent": forwarded_instruction,
                    "contract": contract,
                }

                # Run agent
                response = agent.run(json.dumps(step_input, indent=2))

                # Persist memory
                self._add_to_agent_memory(agent.name, loop_cycle, response)
                self._add_memory(agent.name, response)

                # Parse additional_instruction if JSON OK
                try:
                    parsed = json.loads(response)
                    forwarded_instruction = parsed.get("additional_instruction", "")
                except Exception:
                    forwarded_instruction = None

                # TERMINATION CHECK
                if self.check_termination(response, has_exit_privilege):
                    print(f"[LOOP TERMINATED] Agent '{agent.name}' triggered termination.")
                    self.loop_flag = False
                    break

        # Return last output
        if self.long_memory:
            return self.long_memory[-1]["output"]
        return ""

    # ----------------------------------------
    # Utilities
    # ----------------------------------------
    def get_agent_memory(self, agent_name: str) -> List[Dict[str, Any]]:
        return self.agent_persistent_memory.get(agent_name, [])

    def clear_agent_memory(self, agent_name: str = None):
        if agent_name:
            self.agent_persistent_memory[agent_name] = []
        else:
            self.agent_persistent_memory = {agent.name: [] for agent in self.agents}

    def exit_loop(self):
        self.loop_flag = False
