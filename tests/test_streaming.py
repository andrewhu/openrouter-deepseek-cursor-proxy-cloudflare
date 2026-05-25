from __future__ import annotations

import unittest

from deepseek_cursor_proxy.streaming import (
    CursorReasoningDisplayAdapter,
    MAX_ACCUMULATED_FIELD_CHARS,
    StreamAccumulator,
    fold_reasoning_into_content,
)


class StreamAccumulatorCapTests(unittest.TestCase):
    def test_content_accumulation_is_capped(self) -> None:
        accumulator = StreamAccumulator()
        chunk_size = 1024
        chunks_needed = (MAX_ACCUMULATED_FIELD_CHARS // chunk_size) + 10
        with self.assertLogs("deepseek_cursor_proxy", level="WARNING") as logs:
            for _ in range(chunks_needed):
                accumulator.ingest_chunk(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "x" * chunk_size},
                            }
                        ]
                    }
                )
        choice = accumulator.choices[0]
        self.assertLessEqual(len(choice.content), MAX_ACCUMULATED_FIELD_CHARS)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("field=content", logs.output[0])

    def test_reasoning_content_accumulation_is_capped(self) -> None:
        accumulator = StreamAccumulator()
        chunk_size = 1024
        chunks_needed = (MAX_ACCUMULATED_FIELD_CHARS // chunk_size) + 10
        with self.assertLogs("deepseek_cursor_proxy", level="WARNING") as logs:
            for _ in range(chunks_needed):
                accumulator.ingest_chunk(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": "r" * chunk_size},
                            }
                        ]
                    }
                )
        choice = accumulator.choices[0]
        self.assertTrue(choice.has_reasoning_content)
        self.assertLessEqual(len(choice.reasoning_content), MAX_ACCUMULATED_FIELD_CHARS)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("field=reasoning_content", logs.output[0])


class CursorReasoningDisplayAdapterTests(unittest.TestCase):
    def test_mirrors_reasoning_content_into_details_content(self) -> None:
        adapter = CursorReasoningDisplayAdapter()
        reasoning_chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "Need context."},
                    "finish_reason": None,
                }
            ],
        }
        answer_chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Final answer."},
                    "finish_reason": None,
                }
            ],
        }

        adapter.rewrite_chunk(reasoning_chunk)
        adapter.rewrite_chunk(answer_chunk)

        self.assertEqual(
            reasoning_chunk["choices"][0]["delta"]["content"],
            "<details>\n<summary>Thinking</summary>\n\nNeed context.",
        )
        self.assertEqual(answer_chunk["choices"][0]["delta"]["content"], "\n</details>\n\nFinal answer.")

    def test_closes_thinking_block_before_tool_calls(self) -> None:
        adapter = CursorReasoningDisplayAdapter()
        adapter.rewrite_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "Need a tool."},
                    }
                ]
            }
        )
        tool_chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ]
        }

        adapter.rewrite_chunk(tool_chunk)

        self.assertEqual(tool_chunk["choices"][0]["delta"]["content"], "\n</details>\n\n")

    def test_flush_chunk_closes_unfinished_thinking_block_at_done(self) -> None:
        adapter = CursorReasoningDisplayAdapter()
        adapter.rewrite_chunk(
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "Still thinking."},
                    }
                ],
            }
        )

        closing_chunk = adapter.flush_chunk("deepseek-v4-pro")

        self.assertIsNotNone(closing_chunk)
        assert closing_chunk is not None
        self.assertEqual(closing_chunk["id"], "chatcmpl-stream")
        self.assertEqual(closing_chunk["model"], "deepseek-v4-pro")
        self.assertEqual(closing_chunk["choices"][0]["delta"]["content"], "\n</details>\n\n")
        self.assertIsNone(adapter.flush_chunk("deepseek-v4-pro"))


class FoldReasoningTests(unittest.TestCase):
    def test_fold_reasoning_into_non_streaming_content(self) -> None:
        payload = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "thinking",
                    },
                }
            ]
        }

        fold_reasoning_into_content(payload, collapsible=True)

        self.assertEqual(
            payload["choices"][0]["message"]["content"],
            "<details>\n<summary>Thinking</summary>\n\nthinking\n</details>\n\nanswer",
        )

    def test_fold_reasoning_skips_empty_reasoning(self) -> None:
        payload = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "",
                    },
                }
            ]
        }

        fold_reasoning_into_content(payload, collapsible=True)

        self.assertEqual(payload["choices"][0]["message"]["content"], "answer")
