"""Ask an LLM to design an RC low-pass filter using the spice-agent MCP server.

Usage:
    python agent.py "Design a low-pass filter with a 2.5 kHz cutoff using a 10 nF
    capacitor."
"""
import asyncio
import sys

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.messages import BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel


DEFAULT_MODEL = "google_genai:gemini-3.8-flash"
#DEFAULT_MODEL = "google_genai:gemini-3.5-flash-lite"

PROMPT_VERSION = 'v2'

SYSTEM_PROMPT = (
    "You are an analog design assistant. Use the simulate_rc_lowpass tool to "
    "check every design you propose. Iterate until the simulated cutoff is "
    "within 2% of the requested value, then report R, C, and the simulated cutoff. "
    "Prefer standard values: E24 or E96 resistors and E12 capacitors."
    "Never return a design that misses the 2% target; if you cannot meet it, say so."
    "Only call a value 'simulated' if you ran the tool with it in this conversation; "
    "otherwise say 'calculated'. Answer in plain text, without LaTeX."
)


class FilterDesign(BaseModel):
    r_ohms: float
    c_farads: float
    simulated_cutoff_hz: float
    notes: str

def should_retry(error: Exception) -> bool:
    """Retry server hiccups (503), but not billing (402), bad model name (404) or quota (429)."""
    return not any(code in str(error) for code in ("402", "404", "429"))

async def run_agent(
    request: str,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
) -> tuple[FilterDesign, list[BaseMessage]]:
    client = MultiServerMCPClient(
        {
            "spice-agent": {
                "command": sys.executable,
                "args": ["-m", "spice_agent.server"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    middleware = [
        ModelRetryMiddleware(
            max_retries=4,
            initial_delay=5.0,
            on_failure="error",
            retry_on=should_retry,
),
    ]
    if fallback_model is not None:
        middleware.append(ModelFallbackMiddleware(fallback_model))

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=FilterDesign,
        middleware=middleware,
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": request}]})
    design = FilterDesign.model_validate(result["structured_response"])
    return design, result["messages"]


async def main(request: str, model: str = DEFAULT_MODEL) -> None:
    design, messages = await run_agent(request, model)
    print(
        f"\nStructured result: R = {design.r_ohms:g} ohms, "
        f"C = {design.c_farads:g} F, cutoff = {design.simulated_cutoff_hz:g} Hz"
    )

    for message in messages:
        if message.type == "ai" and message.tool_calls:
            for call in message.tool_calls:
                print(f"-> {call['name']}({call['args']})")
        elif message.type == "tool":
            print(f"   {message.text}")
        elif message.type == "ai":
            print("\nFinal answer:\n" + message.text)


if __name__ == "__main__":
    asyncio.run(
        main(
            " ".join(sys.argv[1:])
            or "Design a low-pass filter with a 2.5 kHz cutoff using a 10 nF capacitor."
        )
    )
