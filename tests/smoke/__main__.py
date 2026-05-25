"""Run OpenRouter smoke tests: python -m tests.smoke"""

from __future__ import annotations

import unittest

if __name__ == "__main__":
    unittest.main(module="tests.smoke.openrouter", verbosity=2)
