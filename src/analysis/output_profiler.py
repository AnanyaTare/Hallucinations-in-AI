"""
output_profiler.py — Run prompts against LLMs and profile their outputs.
Generates raw evaluation data stored in data/evaluation/.
"""

import asyncio
from loguru import logger
from typing import List, Optional
from datetime import datetime, timezone

from src.utils.api_client import query_llm
from src.utils.data_loader import save_evaluation_outputs


async def _run_single(
    prompt: str,
    model: str,
    mitigation: str,
    ground_truth: Optional[str],
    index: int,
    total: int,
) -> dict:
    """Run one prompt against the model and return a result dict."""
    logger.info(f"[profiler] [{index+1}/{total}] model={model} | prompt='{prompt[:50]}...'")
    try:
        response = await query_llm(prompt=prompt, model=model, temperature=0.0)
        status = "success"
    except Exception as e:
        response = ""
        status = f"error: {e}"
        logger.error(f"[profiler] LLM call failed: {e}")

    return {
        "index": index,
        "prompt": prompt,
        "response": response,
        "ground_truth": ground_truth,
        "model": model,
        "mitigation": mitigation,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def profile_outputs_async(
    prompts: List[str],
    model: str = "gpt-4o",
    mitigation: str = "none",
    ground_truths: Optional[List[str]] = None,
    save_filename: str = "model_outputs.jsonl",
    concurrency: int = 3,
) -> List[dict]:
    """
    Run all prompts concurrently (with concurrency limit) and save results.

    Args:
        prompts: List of question strings to send to the LLM.
        model: LLM model identifier.
        mitigation: Mitigation technique applied (for metadata).
        ground_truths: Optional list of reference answers (same length as prompts).
        save_filename: Output filename in data/evaluation/.
        concurrency: Max simultaneous LLM calls.

    Returns:
        List of result dicts.
    """
    ground_truths = ground_truths or [None] * len(prompts)
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(i, prompt, gt):
        async with semaphore:
            return await _run_single(prompt, model, mitigation, gt, i, len(prompts))

    tasks = [bounded(i, p, gt) for i, (p, gt) in enumerate(zip(prompts, ground_truths))]
    results = await asyncio.gather(*tasks)

    save_evaluation_outputs(list(results), filename=save_filename)
    logger.success(f"[profiler] Saved {len(results)} outputs → data/evaluation/{save_filename}")
    return list(results)


def profile_outputs(
    prompts: List[str],
    model: str = "gpt-4o",
    mitigation: str = "none",
    ground_truths: Optional[List[str]] = None,
    save_filename: str = "model_outputs.jsonl",
    concurrency: int = 3,
) -> List[dict]:
    """Synchronous wrapper for profile_outputs_async. Use from scripts/notebooks."""
    return asyncio.run(
        profile_outputs_async(prompts, model, mitigation, ground_truths, save_filename, concurrency)
    )
