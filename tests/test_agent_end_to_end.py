"""Run the real agent loop, MCP server and ngspice with a scripted stand-in model.

No API calls: the model is replaced by a fake that replays fixed tool calls.
"""
import asyncio
import shutil
import unittest
from itertools import count
from unittest.mock import patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent import AgentStoppedEarly, run_agent


class ScriptedModel(BaseChatModel):
    """Returns the next scripted tool call each time it is asked; repeats the last one."""

    script: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        step = sum(1 for message in messages if message.type == "ai")
        name, args = self.script[min(step, len(self.script) - 1)]
        if name == "loop":                       # keep simulating new values forever
            name, args = "simulate_rc_lowpass", {"r_ohms": 6000 + 10 * step, "c_farads": 1e-8}
        message = AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call-{step}"}],
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


@unittest.skipUnless(shutil.which("ngspice"), "ngspice is not installed")
class AgentEndToEndTests(unittest.TestCase):
    def test_lookup_then_simulate_then_answer(self) -> None:
        model = ScriptedModel(script=[
            ("find_standard_values", {"target_cutoff_hz": 2500, "fixed_c_farads": 1e-8}),
            ("simulate_rc_lowpass", {"r_ohms": 6340, "c_farads": 1e-8}),
            ("FilterDesign", {"r_ohms": 6340, "c_farads": 1e-8,
                              "simulated_cutoff_hz": 2510.33, "notes": "E96 resistor."}),
        ])
        with patch("agent.build_chat_model", return_value=model):
            design, messages = asyncio.run(run_agent("2.5 kHz with 10 nF", "scripted"))
        self.assertEqual((design.r_ohms, design.c_farads), (6340, 1e-8))
        tool_results = [m.text for m in messages if m.type == "tool"]
        self.assertIn("10 nF", tool_results[0])                 # lookup result
        self.assertIn("Measured cutoff: 2510.33 Hz", tool_results[1])  # real ngspice run

    def test_runaway_is_stopped_with_its_trace(self) -> None:
        model = ScriptedModel(script=[("loop", {})])
        with patch("agent.build_chat_model", return_value=model):
            with self.assertRaises(AgentStoppedEarly) as caught:
                asyncio.run(run_agent("loop forever", "scripted", max_model_calls=4))
        simulated = [
            call["args"]["r_ohms"]
            for message in caught.exception.messages
            if message.type == "ai"
            for call in message.tool_calls
        ]
        self.assertEqual(caught.exception.reason, "model call limit reached")
        self.assertEqual(simulated, [6000, 6010, 6020, 6030])   # 4 calls, then stopped


if __name__ == "__main__":
    unittest.main()
