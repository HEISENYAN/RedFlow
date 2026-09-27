from __future__ import annotations

VALUE_PROMPT_TEMPLATE = """You are estimating robot task progress.

Task: {instruction}

Observation: <image>
"""


PAIR_PROMPT_TEMPLATE = """You are estimating robot task progress.

Task: {instruction}

Current observation: <image>

Historical observation: <image>
"""
