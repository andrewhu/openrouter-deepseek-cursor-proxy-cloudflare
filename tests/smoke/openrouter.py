"""OpenRouter smoke tests — run manually with a real API key (billed usage).

Not discovered by default unittest (filename has no test_ prefix).
Run: RUN_LIVE_OPENROUTER_TESTS=1 LIVE_OPENROUTER_KEY=sk-or-... uv run smoke
"""

from __future__ import annotations

from copy import deepcopy
import unittest

from deepseek_cursor_proxy.config import OPENROUTER_MODEL_ID

from tests.support.live import (
    LiveProxyFixture,
    live_get,
    live_openrouter_key,
    live_post_json,
    skip_unless_live_openrouter,
)


def tool_call_first_request() -> dict:
    return {
        "model": "deepseek-v4-pro",
        "messages": [
            {
                "role": "user",
                "content": ("Use the get_date tool exactly once, " "then tell me the date it returns."),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_date",
                    "description": "Return the current date as YYYY-MM-DD.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "required",
    }


@skip_unless_live_openrouter
class OpenRouterSmokeTests(unittest.TestCase):
    api_key: str
    proxy: LiveProxyFixture

    @classmethod
    def setUpClass(cls) -> None:
        key = live_openrouter_key()
        assert key is not None
        cls.api_key = key

    def setUp(self) -> None:
        self.proxy = LiveProxyFixture(self.api_key).start()

    def tearDown(self) -> None:
        self.proxy.close()

    def test_healthz_and_models_with_valid_bearer(self) -> None:
        health_status, health_body = live_get(
            f"{self.proxy.base_url}/healthz",
            api_key=self.api_key,
        )
        self.assertEqual(health_status, 200)
        self.assertTrue(health_body.get("ok"))

        models_status, models_body = live_get(
            f"{self.proxy.base_url}/v1/models",
            api_key=self.api_key,
        )
        self.assertEqual(models_status, 200)
        self.assertTrue(models_body.get("data"))

    def test_rejects_wrong_bearer(self) -> None:
        status, body = live_post_json(
            self.proxy.chat_completions_url,
            {
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "hi"}],
            },
            api_key="sk-wrong-key",
        )
        self.assertEqual(status, 401)
        self.assertIn("error", body)

    def test_simple_chat_completion(self) -> None:
        status, response = live_post_json(
            self.proxy.chat_completions_url,
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly: PONG",
                    }
                ],
            },
            api_key=self.api_key,
        )
        self.assertEqual(status, 200, response.get("error"))
        message = response["choices"][0]["message"]
        self.assertTrue(message.get("content"))

    def test_proxy_repairs_real_openrouter_tool_call_history(self) -> None:
        first_status, first_response = live_post_json(
            self.proxy.chat_completions_url,
            tool_call_first_request(),
            api_key=self.api_key,
        )
        self.assertEqual(first_status, 200, first_response.get("error"))
        assistant_with_reasoning = first_response["choices"][0]["message"]
        self.assertTrue(assistant_with_reasoning.get("reasoning_content") or assistant_with_reasoning.get("reasoning"))
        self.assertTrue(assistant_with_reasoning.get("tool_calls"))

        cursor_assistant = deepcopy(assistant_with_reasoning)
        cursor_assistant.pop("reasoning_content", None)
        cursor_assistant.pop("reasoning", None)
        tool_messages = [
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": "2026-04-24",
            }
            for tool_call in cursor_assistant["tool_calls"]
        ]
        missing_reasoning_payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                tool_call_first_request()["messages"][0],
                cursor_assistant,
                *tool_messages,
            ],
            "tools": tool_call_first_request()["tools"],
        }

        direct_status, direct_response = live_post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {
                **missing_reasoning_payload,
                "model": OPENROUTER_MODEL_ID,
                "reasoning": {"effort": "xhigh"},
                "provider": {
                    "only": ["deepseek"],
                    "allow_fallbacks": False,
                },
            },
            api_key=self.api_key,
        )
        self.assertEqual(direct_status, 400)
        self.assertIn(
            "reasoning",
            direct_response.get("error", {}).get("message", "").lower(),
        )

        proxy_status, second_response = live_post_json(
            self.proxy.chat_completions_url,
            missing_reasoning_payload,
            api_key=self.api_key,
        )
        self.assertEqual(proxy_status, 200, second_response.get("error"))
        final_assistant = second_response["choices"][0]["message"]
        self.assertTrue(final_assistant.get("content") or final_assistant.get("tool_calls"))

        if final_assistant.get("content"):
            cursor_final = deepcopy(final_assistant)
            cursor_final.pop("reasoning_content", None)
            cursor_final.pop("reasoning", None)
            followup_payload = {
                "model": "deepseek-v4-pro",
                "messages": [
                    tool_call_first_request()["messages"][0],
                    cursor_assistant,
                    *tool_messages,
                    cursor_final,
                    {"role": "user", "content": "Reply with exactly: OK"},
                ],
                "tools": tool_call_first_request()["tools"],
            }
            followup_status, followup_response = live_post_json(
                self.proxy.chat_completions_url,
                followup_payload,
                api_key=self.api_key,
            )
            self.assertEqual(followup_status, 200, followup_response.get("error"))


if __name__ == "__main__":
    unittest.main()
