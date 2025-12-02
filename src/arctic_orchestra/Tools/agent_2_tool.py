class Agent2Tool:
    """
    Wraps an Agent instance into a fully compliant tool callable.

    This wrapper converts an existing Agent into a tool that can be consumed by
    other Agents within the Arctic-Orchestra framework. The generated tool:

    - Exposes a stable, schema-friendly signature with a single `input` parameter.
    - Uses the underlying Agent's name as the tool name by default.
    - Provides a `__describe__` method for automatic schema extraction.
    - Includes optional additional prompt text that is appended to the tool's
    description to influence downstream LLM behavior.

    The resulting callable integrates seamlessly with the Agent tool system,
    allowing hierarchical or recursive orchestration where Agents can delegate
    sub-tasks to other Agents as tools.

    Parameters
    ----------
    agent : Agent
        The Agent instance to wrap as a tool. All calls to the tool will be
        routed to `agent.run(...)`.

    additional_prompt : str, optional
        Extra natural-language guidance appended to the tool's description.
        Useful for overriding defaults or injecting task-specific hints into
        the tool metadata provided to the LLM. Default is an empty string.

    Returns
    -------
    Tuple[str, Callable]
        A tuple containing:
        - the tool name (derived from agent.name)
        - the generated tool function with a stable interface and __describe__()
    """
    def __init__(self, agent, additional_prompt: str = ""):
        self.agent = agent
        self.tool_name = agent.name.replace(" ", "_").lower()
        self.additional_prompt = additional_prompt

    def create_tool(self):
        agent = self.agent
        tool_name = self.tool_name
        extra = f"\nAdditional Instructions: {self.additional_prompt}" if self.additional_prompt else ""

        # -------- Build tool description --------
        description_text = (
            f"Delegates a sub-task to agent '{agent.name}'.\n"
            f"Identity: {agent.identity}\n"
            f"Task: {agent.task}\n"
            f"{extra}"
        )

        usage_example = (
            f'{{"tool": "{tool_name}", "args": {{"input": "your query"}}}}'
        )

        # -------- The actual tool callable --------
        def agent_tool(input: str = None):
            # Describe mode
            if input == "__describe__":
                return (description_text, usage_example)

            # Normal mode → run the agent
            return agent.run(input)

        # provide describe method
        def __describe__():
            return description_text, usage_example

        agent_tool.__describe__ = __describe__

        # parameter signature
        agent_tool.__annotations__ = {"input": str}

        return tool_name, agent_tool
