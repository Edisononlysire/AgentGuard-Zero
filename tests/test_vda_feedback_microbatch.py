from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vda_feedback_server import FeedbackEngine


class FeedbackGenerationMicrobatchTests(unittest.TestCase):
    def _engine(self, batch_size: int):
        engine = object.__new__(FeedbackEngine)
        engine.args = SimpleNamespace(generation_batch_size=batch_size)
        calls: list[list[str]] = []

        def generate(batch):
            values = [messages[0]["content"] for messages in batch]
            calls.append(values)
            return [f"output:{value}" for value in values]

        engine._generate_batch = generate
        return engine, calls

    def test_chunks_preserve_all_prompts_and_order(self) -> None:
        engine, calls = self._engine(2)
        prompts = [[{"role": "user", "content": str(index)}] for index in range(5)]

        outputs = engine._generate_in_chunks(prompts)

        self.assertEqual(calls, [["0", "1"], ["2", "3"], ["4"]])
        self.assertEqual(outputs, [f"output:{index}" for index in range(5)])

    def test_zero_keeps_legacy_single_batch(self) -> None:
        engine, calls = self._engine(0)
        prompts = [[{"role": "user", "content": str(index)}] for index in range(3)]

        outputs = engine._generate_in_chunks(prompts)

        self.assertEqual(calls, [["0", "1", "2"]])
        self.assertEqual(outputs, ["output:0", "output:1", "output:2"])


if __name__ == "__main__":
    unittest.main()
