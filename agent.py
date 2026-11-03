"""Ask an LLM to design an RC low-pass filter using the spice-agent MCP server.

Usage:
    python agent.py "Design a low-pass filter with a 2.5 kHz cutoff using a 10 nF capacitor."
    python agent.py --model google_genai:gemini-3.8-flash "Design a ..."

Model names use the form "provider:model":
    ollama:<name>          a local model served by Ollama (no API cost)
    google_genai:<name>    Gemini through Google AI Studio (needs GOOGLE_API_KEY)
"""
import argparse
import asyncio
import sys

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

DEFAULT_MODEL = "google_genai:gemini-3.8-flash"
#DEFAULT_MODEL = "ollama:qwen2.5:7b"
# Cloud option: DEFAULT_MODEL = "google_genai:gemini-3.8-flash"

# Change PROMPT_VERSION whenever SYSTEM_PROMPT or the tools change, so results stay comparable.
# v2.1 = v2 with the missing spaces between sentences fixed.
# v3   = adds the find_standard_values tool; the agent no longer searches parts itself.
# v3.1 = agent reports meets_spec; handles fixed R and C, other tolerances, and limits.
# v3.2 = prompt unchanged; simulate_rc_lowpass now leads with VERIFIED / NOT VERIFIED
#        so an unmeasurable cutoff cannot be mistaken for a simulated one.
# v3.3 = simulated_cutoff_hz may be null ("not simulated") instead of a placeholder 0.
# v3.4 = never relax the requested tolerance or constraints (Flash-Lite once "relaxed"
#        2% to 5% on its own and reported success).
# v3.5 = find_standard_values labels each option ONE / TWO capacitors and always
#        offers a one-capacitor choice; prompt maps "single"/"one of each" to
#        allow_parallel_capacitors=false (v3.4 Flash-Lite used parallel parts anyway).
# v3.6 = v3.5 without the v3.4 "never relax the tolerance" sentence: 10-run tests showed
#        it caused the part-count failures (v3.3 40/40 vs v3.4 28/40 on infeasible cases).
PROMPT_VERSION = "v3.6"

SYSTEM_PROMPT = (
    "You are an analog design assistant for RC low-pass filters. "
    "Step 1: call find_standard_values with the target cutoff, any resistor or "
    "capacitor the request fixes, the tolerance if the request gives one "
    "(tolerance_percent; otherwise 2), any series the request requires (for "
    "example 'only E12 values' means resistor_series='E12' and "
    "capacitor_series='E12'), and whether more than one capacitor is allowed "
    "('a single capacitor' or 'one of each' means allow_parallel_capacitors=false). "
    "If the request fixes both R and C, there is nothing to choose: do not change "
    "them, simulate them, and report whether they meet the spec. "
    "Step 2: pick the first option that satisfies every limit in the request (for "
    "example a maximum resistor value, or ONE capacitor when only one is allowed), "
    "and simulate exactly those r_ohms and "
    "c_farads with simulate_rc_lowpass. If no listed option satisfies the limits, "
    "call find_standard_values again with a fixed value inside the limits. "
    "Do not search for part combinations yourself. "
    "If no design can meet the request, simulate the closest option that still "
    "satisfies the request's other constraints (such as one capacitor only), and "
    "clearly state that the request cannot be met. "

    "Set meets_spec to true only if the simulated cutoff is within the requested "
    "tolerance and every constraint in the request is satisfied; otherwise false. "
    "Report R, C, and the simulated cutoff; for parallel capacitors, name both parts. "
    "Only call a value 'simulated' if you ran simulate_rc_lowpass with it in this "
    "conversation; otherwise say 'calculated'. Answer in plain text, without LaTeX."
)

# Cost and runaway protection.
MAX_MODEL_CALLS = 8          # model calls allowed per request; good runs use 2-4
MAX_OUTPUT_TOKENS = 2048     # cap on each model response (Gemini: thinking included)
MODEL_CALL_TIMEOUT_S = 120   # seconds per model call (local CPU models are slow)
LOCAL_CONTEXT_TOKENS = 8192  # Ollama's default context is too small for the tools + history


class FilterDesign(BaseModel):
    r_ohms: float
    c_farads: float
    simulated_cutoff_hz: float | None = Field(
        description="The cutoff measured by simulate_rc_lowpass for the final design, "
        "or null if the simulation could not measure it. Never put a calculated value here."
    )
    meets_spec: bool = Field(
        description="True only if the simulated cutoff is within the requested "
        "tolerance and every constraint in the request is met."
    )
    notes: str


class AgentStoppedEarly(Exception):
    """The agent ended without a final design (for example, it hit the model-call limit).

    Carries the conversation so far, so an evaluation can record what the agent tried.
    """

    def __init__(self, reason: str, messages: list[BaseMessage]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.messages = messages


# Errors that retrying cannot fix: billing (402), bad model name (404), quota (429),
# and a local model that cannot call tools.
NON_RETRYABLE_MARKERS = ("402", "404", "429", "does not support tools")


def should_retry(error: Exception) -> bool:
    """Retry temporary problems such as a 503 'high demand', but nothing in NON_RETRYABLE_MARKERS."""
    message = str(error)
    return not any(marker in message for marker in NON_RETRYABLE_MARKERS)


def build_chat_model(model: str) -> BaseChatModel | str:
    """Turn a "provider:model" name into a chat model with cost limits applied.

    Unrecognized providers are passed through unchanged for LangChain to handle.
    """
    provider, _, name = model.partition(":")
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=name,
            temperature=0,                    # repeatable answers for evaluation
            num_ctx=LOCAL_CONTEXT_TOKENS,
            num_predict=MAX_OUTPUT_TOKENS,
            client_kwargs={"timeout": MODEL_CALL_TIMEOUT_S},
        )
    if provider == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=name,
            thinking_level="low",             # hidden thinking tokens are billed; keep them low
            max_output_tokens=MAX_OUTPUT_TOKENS,
            timeout=MODEL_CALL_TIMEOUT_S,
            max_retries=1,                    # our retry middleware handles retries
        )
    return model


async def run_agent(
    request: str,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
    max_model_calls: int = MAX_MODEL_CALLS,
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
        # Hard stop on runaway loops. "end" stops the agent but keeps the conversation,
        # so we can see what it was trying; run_agent then raises AgentStoppedEarly.
        ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end"),
        ModelRetryMiddleware(
            max_retries=2,
            initial_delay=5.0,
            on_failure="error",
            retry_on=should_retry,
        ),
    ]
    if fallback_model is not None:
        middleware.append(ModelFallbackMiddleware(build_chat_model(fallback_model)))

    agent = create_agent(
        model=build_chat_model(model),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=FilterDesign,
        middleware=middleware,
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": request}]})
    messages = result["messages"]
    if result.get("structured_response") is None:
        last_text = messages[-1].text if messages else ""
        reason = (
            "model call limit reached"
            if "limit" in last_text.lower()
            else "agent ended without a final design"
        )
        raise AgentStoppedEarly(reason, messages)
    design = FilterDesign.model_validate(result["structured_response"])
    return design, messages


async def main(request: str, model: str = DEFAULT_MODEL) -> None:
    design, messages = await run_agent(request, model)
    print(
        f"\nStructured result: R = {design.r_ohms:g} ohms, "
        f"C = {design.c_farads:g} F, simulated cutoff = "
        f"{'not measured' if design.simulated_cutoff_hz is None else f'{design.simulated_cutoff_hz:g} Hz'}"
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
    parser = argparse.ArgumentParser(description="Design an RC low-pass filter with an AI agent.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="provider:model name")
    parser.add_argument("request", nargs="*", help="the design request")
    args = parser.parse_args()
    asyncio.run(
        main(
            " ".join(args.request)
            or "Design a low-pass filter with a 2.5 kHz cutoff using a 10 nF capacitor.",
            args.model,
        )
    )
