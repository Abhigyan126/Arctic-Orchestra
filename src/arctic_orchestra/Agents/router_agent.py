from typing import List, Dict, Optional, Any
from arctic_orchestra.Tools.agent_2_tool import Agent2Tool
from arctic_orchestra.Agents.base import Agent


class RoutingAgent:
    """
    High-level orchestrator that converts multiple specialized Agents
    into callable tools, enforces an execution sequence, and builds a 
    RouterAgent responsible for end-to-end reasoning and routing.
    """

    BASE_ROUTING_CONTRACT = """
    You are a Routing Orchestrator.

    Your job is:
    1. To analyze the user's raw request.
    2. To call the wrapped agent-tools in the correct sequence.
    3. After each tool call, take its **output** and feed it into the next tool directly or by modifying it to its need.
    4. Make sure the input to the agents are tailored to its needs do not feed references of other agents or task that does not relate to the tool.
    5. Continue until all tools have run OR until the task is complete.
    6. Produce the final combined result only after running the last tool.
    """

    def __init__(
        self,
        model_name: str,
        agents: List[Any],
        additional_routing_instructions: Optional[str] = None,
        api_key: Optional[str] = None,
        websearch_config: Optional[Dict] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        debug: bool = False
    ):
        """
        :param model_name: The string identifier for the LLM (e.g., 'gpt-4o').
        :param agents: Ordered list of Agent instances.
        :param additional_routing_instructions: Optional user-defined routing logic.
        :param api_key: Optional API key override.
        :param websearch_config: Configuration for web search if supported.
        :param temperature: LLM creativity (default 1.0).
        :param top_p: LLM nucleus sampling (default 1.0).
        :param debug: Enable debug logging for the router.
        """
        self.model_name = model_name
        self.agents = agents
        self.additional_routing_instructions = additional_routing_instructions or ""
        self.api_key = api_key
        self.websearch_config = websearch_config
        self.temperature = temperature
        self.top_p = top_p
        self.debug = debug
        
        self.router_tools: Dict[str, Any] = {}

    def wrap_agents_as_tools(self):
        """
        Converts each agent into a tool using Agent2Tool.
        Preserves order of the agent list.
        """
        for agent in self.agents:
            tool_name, tool_func = Agent2Tool(
                agent,
                additional_prompt=f"This tool represents: {agent.name}. "
                                  f"Use it only for its specialized purpose."
            ).create_tool()

            self.router_tools[tool_name] = tool_func

        return self.router_tools

    def build_router_agent(self, name: str = "RouterAgent") -> Agent:
        """
        Constructs the final RouterAgent responsible for orchestrating
        all agent-tools in the given order.
        """
        # Step 1 — wrap agents
        router_tools = self.wrap_agents_as_tools()

        # Step 2 — create the final routing instructions
        ordered_tool_list = list(router_tools.keys())

        ordered_instructions = "\n".join(
            [f"{i+1}. Call **{tool_name}**" for i, tool_name in enumerate(ordered_tool_list)]
        )

        routing_prompt = (
            self.BASE_ROUTING_CONTRACT
            + "\nExecution Order:\n"
            + ordered_instructions
            + "\n\nUser Additional Instructions:\n"
            + self.additional_routing_instructions
        )

        # Step 3 — construct Router Agent with all config params
        router_agent = Agent(
            model_name=self.model_name,
            name=name,
            identity="You are the master router coordinating specialized agents.",
            instruction=routing_prompt,
            tools=router_tools,
            api_key=self.api_key,
            websearch_config=self.websearch_config,
            temperature=self.temperature,
            top_p=self.top_p,
            debug=self.debug
        )

        return router_agent