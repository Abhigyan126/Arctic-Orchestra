import json
from typing import List, Dict, Any
from sequential_agent import SequentialAgent


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
        # Persistent local memory per agent across loops
        self.agent_persistent_memory = {agent.name: [] for agent in agents}
        # Loop control flag - can be set to False via terminate_loop flag in agent output
        self.loop_flag = True
        # List of agents that are allowed to terminate the loop
        self.agents_with_exit_flag = agents_with_exit_flag or []

    def _add_to_agent_memory(self, agent_name: str, loop_cycle: int, output: str):
        """Add to agent's persistent local memory with sliding window."""
        entry = {"loop_cycle": loop_cycle, "output": output}
        self.agent_persistent_memory[agent_name].append(entry)
        
        # Enforce sliding window on local memory
        if len(self.agent_persistent_memory[agent_name]) > self.local_memory_window:
            self.agent_persistent_memory[agent_name].pop(0)

    def _build_contract(self, agent, has_exit_privilege: bool) -> str:
        """Build the output contract based on whether agent can terminate loop."""
        base_contract = (
            "When producing your final answer, YOU MUST output strict JSON:\n"
            "{\n"
            '   "final_output": "<your main generated content>",\n'
            '   "additional_instruction": "<optional guidance for next agent or empty string>"'
        )
        
        if has_exit_privilege:
            exit_contract = (
                ',\n'
                '   "terminate_loop": <true or false>\n'
                "}\n\n"
                "Rules:\n"
                "- Do NOT include any explanation outside the JSON structure.\n"
                "- `final_output` contains your main response.\n"
                "- `additional_instruction` passes context to the next agent (can be empty).\n"
                "- `terminate_loop` should be set to true ONLY when the entire workflow is complete and it satisfies the provided task.\n"
                "- Set `terminate_loop` to false if the workflow should continue to the next cycle.\n"
                "- Your local memory shows what YOU did in previous loop cycles.\n"
                "- Global memory shows what ALL agents have done.\n"
                "- Incorporate any forwarded instruction from the previous agent.\n"
            )
        else:
            exit_contract = (
                '\n'
                "}\n\n"
                "Rules:\n"
                "- Do NOT include any explanation outside the JSON structure.\n"
                "- `final_output` contains your main response.\n"
                "- `additional_instruction` passes context to the next agent (can be empty).\n"
                "- Your local memory shows what YOU did in previous loop cycles.\n"
                "- Global memory shows what ALL agents have done.\n"
                "- Incorporate any forwarded instruction from the previous agent.\n"
            )
        
        return base_contract + exit_contract

    def run(self, user_query: str) -> str:
        """
        Run the sequential agent workflow with looping capability.
        Flow: Agent1 → Agent2 → Agent3 → Agent1 → Agent2 → Agent3 → ...
        Agents in agents_with_exit_flag list can terminate the loop via terminate_loop flag.
        """
        # Reset global memory for this run
        self.long_memory = []
        self.short_memory = []
        
        # Reset all agents' persistent memory at start
        self.agent_persistent_memory = {agent.name: [] for agent in self.agents}
        
        # Reset loop flag
        self.loop_flag = True
        
        forwarded_instruction = None
        loop_cycle = 0

        # Main loop: keeps cycling through all agents until terminate_loop flag or max_loops
        while self.loop_flag and loop_cycle < self.max_loops:
            loop_cycle += 1
            
            # Iterate through each agent in the sequence
            for step_index, agent in enumerate(self.agents, start=1):
                
                # Check if loop was terminated by previous agent
                if not self.loop_flag:
                    break

                # Check if this agent has exit privilege
                has_exit_privilege = agent in self.agents_with_exit_flag
                
                # Build the contract with or without termination flag
                contract = self._build_contract(agent, has_exit_privilege)

                # Construct the input with all memory contexts
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

                # Run the agent
                response = agent.run(json.dumps(step_input, indent=2))

                # Store in agent's persistent local memory
                self._add_to_agent_memory(agent.name, loop_cycle, response)

                # Store in global memory (with compression if needed)
                self._add_memory(agent.name, response)

                # Parse response to extract forwarded instructions and check termination flag
                try:
                    parsed = json.loads(response)
                    forwarded_instruction = parsed.get("additional_instruction", "")
                    
                    # Only check terminate_loop flag if this agent has exit privilege
                    if has_exit_privilege:
                        terminate_flag = parsed.get("terminate_loop", False)
                        if terminate_flag:
                            print(f"[LOOP TERMINATED] Agent '{agent.name}' set terminate_loop=true")
                            self.loop_flag = False
                            break
                            
                except json.JSONDecodeError:
                    forwarded_instruction = None
                    # If JSON parsing fails, continue normally

        # Return the last output from global memory
        if self.long_memory:
            return self.long_memory[-1]["output"]
        return ""

    def get_agent_memory(self, agent_name: str) -> List[Dict[str, Any]]:
        """Retrieve persistent memory for a specific agent."""
        return self.agent_persistent_memory.get(agent_name, [])

    def clear_agent_memory(self, agent_name: str = None):
        """Clear persistent memory for one or all agents."""
        if agent_name:
            self.agent_persistent_memory[agent_name] = []
        else:
            self.agent_persistent_memory = {agent.name: [] for agent in self.agents}

    def exit_loop(self):
        """Manually exit the loop. Can be called externally if needed."""
        self.loop_flag = False