"""
self_consistency.py — Self-consistency sampling mitigation.
Generates N responses and selects the most common answer via majority voting.
"""

import asyncio
from collections import Counter
from loguru import logger
from typing import List
from src.utils.api_client import query_llm


async def _sample_responses(
    prompt: str,
    model: str,
    n: int,
    temperature: float,
) -> List[str]:
    """Generate N diverse responses in parallel."""
    tasks = [query_llm(prompt=prompt, model=model, temperature=temperature) for _ in range(n)]
    return list(await asyncio.gather(*tasks))


def self_consistency_answer(
    prompt: str,
    model: str = "gpt-4o",
    n: int = 5,
    temperature: float = 0.7,
) -> dict:
    """
    Apply self-consistency: generate N responses and select the majority answer.

    Args:
        prompt: The question to ask.
        model: LLM to use.
        n: Number of diverse samples to generate.
        temperature: Sampling temperature (higher = more diverse).

    Returns:
        Dict with 'answer' (winner), 'votes', 'all_responses'.
    """
    responses = asyncio.run(_sample_responses(prompt, model, n, temperature))

    vote_counter = Counter(responses)
    winner, winner_count = vote_counter.most_common(1)[0]

    vote_summary = {
        resp[:100] + ("..." if len(resp) > 100 else ""): count
        for resp, count in vote_counter.most_common()
    }

    logger.info(
        f"[self_consistency] n={n} | winner_votes={winner_count}/{n} | "
        f"unique_answers={len(vote_counter)}"
    )

    return {
        "answer":        winner,
        "vote_count":    winner_count,
        "total_samples": n,
        "votes":         vote_summary,
        "all_responses": responses,
    }
