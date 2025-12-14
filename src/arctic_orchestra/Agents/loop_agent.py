from typing import List, Dict, Any, Optional, Callable
from copy import deepcopy
import json

class LoopAgent:
    """
    LoopAgent: multi-agent loop with selective exit-loop capability.
    Only agents explicitly passed in exit_allowed_agents can call exit_loop.
    """

    def __init__(
        self,
        agents: List[Any],
        name: str = "LoopAgent",
        compression_model: Optional[Callable] = None,
        window_size: int = 2,
        max_context_chars: int = 8000,
        max_iterations: int = 5,
        exit_allowed_agents: Optional[List[Any]] = None,   # now takes AGENT INSTANCES
        exit_instructions: Optional[str] = None,           # new: instructions we inject
        max_history_per_model: int = 20,
        max_model_history_chars: int = 2000,
    ):
        self.name = name
        self.agents = agents
        self.compression_model = compression_model
        self.window_size = window_size
        self.max_context_chars = max_context_chars
        self.max_iterations = max_iterations
        self.exit_instructions = exit_instructions or ""
        self.max_history_per_model = max_history_per_model
        self.max_model_history_chars = max_model_history_chars

        # convert agent instances -> names
        if exit_allowed_agents:
            self.exit_allowed_names = {a.name for a in exit_allowed_agents}
        else:
            self.exit_allowed_names = set()

        self.long_memory = []
        self.short_memory = []
        self.per_model_history = {}

    # ---------------------------
    # exit tool factory
    # ---------------------------
    def _make_exit_tool(self, control):
        def exit_loop(reason: str = None, payload: Any = None) -> str:
            control["should_exit"] = True
            control["reason"] = reason
            control["payload"] = payload
            return json.dumps({
                "status": "exit_requested",
                "reason": reason or "",
                "payload_summary": str(payload)[:500]
            })
        exit_loop.__doc__ = "Tool allowing this agent to exit the loop early."
        return exit_loop

    # ------------------------------------------------
    # Internal helpers (missing pieces filled in)
    # ------------------------------------------------
    def _enforce_per_model_history_limits(self, agent_name: str) -> None:
        """
        Ensure per-model history respects both:
         - max_history_per_model (number of entries)
         - max_model_history_chars (approximate chars across output_preview)
        Removes oldest entries first until constraints satisfied.
        """
        history = self.per_model_history.get(agent_name, [])

        # enforce count limit
        while len(history) > self.max_history_per_model:
            history.pop(0)

        # enforce char limit on cumulative output_preview lengths
        total_chars = sum(len(entry.get("output_preview", "")) for entry in history)
        while history and total_chars > self.max_model_history_chars:
            removed = history.pop(0)
            total_chars -= len(removed.get("output_preview", ""))

        # write back in case
        self.per_model_history[agent_name] = history

    def _add_memory(self, agent_name: str, response: Any) -> None:
        """
        Add a memory entry:
         - short_memory: sliding window of most recent items (kept to window_size)
         - long_memory: append full (or compressed) response for longer-term storage
        If compression_model is provided, use it to compress the response before storing
        in long_memory.
        """
        # Short memory: keep latest window_size items (store small preview)
        preview = str(response)[:1000]
        short_entry = {"agent": agent_name, "preview": preview}
        self.short_memory.append(short_entry)
        # trim short_memory to window_size
        if len(self.short_memory) > self.window_size:
            # remove oldest until at window size
            excess = len(self.short_memory) - self.window_size
            if excess > 0:
                self.short_memory = self.short_memory[excess:]

        # Long memory: store either compressed or raw response
        long_entry = {
            "agent": agent_name,
            "full_output": None,
            "compressed": None
        }
        try:
            if callable(self.compression_model):
                # Compression model expected to return a string summary/compressed form
                compressed = self.compression_model(response)
                long_entry["compressed"] = compressed
                # Keep the raw output only if small (protect memory size)
                long_entry["full_output"] = str(response)[:2000]
            else:
                long_entry["full_output"] = response
        except Exception:
            # if compression fails, just store the raw (safe fallback)
            long_entry["full_output"] = response

        self.long_memory.append(long_entry)

    # ------------------------------------------------
    # Main run loop
    # ------------------------------------------------
    def run(self, user_query: str) -> Dict[str, Any]:

        self.long_memory = []
        self.short_memory = []
        self.per_model_history = {}

        run_log = []
        final_output = ""
        exited = False
        exit_reason = None

        # LOOP
        for iteration in range(1, self.max_iterations + 1):
            for agent in self.agents:

                agent_name = agent.name
                control = {"should_exit": False, "reason": None, "payload": None}

                # ---- prepare per-model history
                self.per_model_history.setdefault(agent_name, [])

                # == SAVE ORIGINAL STATE ==
                original_tools = deepcopy(agent.tools)
                original_instruction = agent.instruction

                # ---------- CONDITIONAL EXIT INJECTION ----------
                allowed = agent_name in self.exit_allowed_names
                injected = False

                if allowed:
                    exit_tool = self._make_exit_tool(control)

                    # inject tool
                    agent.tools = deepcopy(original_tools)
                    agent.tools["exit_loop"] = exit_tool

                    # update schemas
                    if hasattr(agent, "_build_tool_schemas"):
                        agent.tool_schemas = agent._build_tool_schemas()

                    # inject exit instructions
                    if self.exit_instructions:
                        agent.instruction = (
                            original_instruction + "\n\n" +
                            "IMPORTANT: " + self.exit_instructions
                        )

                    injected = True
                else:
                    # make sure they don't see or know exit exists
                    agent.tools = deepcopy(original_tools)
                    agent.instruction = original_instruction

                # ---------- RUN AGENT ----------
                agent_input = {
                    "original_query": user_query,
                    "iteration": iteration,
                    "agent_name": agent_name,
                    "short_memory": self.short_memory,
                    "long_memory": self.long_memory,
                    "per_model_history": self.per_model_history[agent_name],
                }

                try:
                    response = agent.run(json.dumps(agent_input))
                except Exception as e:
                    response = f"Agent execution error: {str(e)}"

                # ---------- RESTORE AGENT CLEANLY ----------
                agent.tools = original_tools
                agent.instruction = original_instruction
                if hasattr(agent, "_build_tool_schemas"):
                    agent.tool_schemas = agent._build_tool_schemas()

                # ---------- MEMORY RECORDING ----------
                model_record = {
                    "iteration": iteration,
                    "agent": agent_name,
                    "input": agent_input,
                    "output": response
                }
                run_log.append(model_record)

                self.per_model_history[agent_name].append({
                    "iteration": iteration,
                    "input_preview": str(agent_input)[:1000],
                    "output_preview": str(response)[:2000],
                })
                self._enforce_per_model_history_limits(agent_name)

                self._add_memory(agent_name, response)

                # ---------- CHECK EXIT ----------
                if allowed and control["should_exit"]:
                    exited = True
                    exit_reason = control["reason"]
                    final_output = (
                        f"Loop exited by agent '{agent_name}' at iteration {iteration}.\n"
                        f"Reason: {exit_reason}\n"
                        f"Last agent response: {response}"
                    )
                    break

            if exited:
                break

        if not exited:
            final_output = run_log[-1]["output"] if run_log else ""

        return {
            "final_output": final_output,
            "exited": exited,
            "exit_reason": exit_reason,
            "run_log": run_log,
            "long_memory": self.long_memory,
            "short_memory": self.short_memory,
            "per_model_history": self.per_model_history
        }
