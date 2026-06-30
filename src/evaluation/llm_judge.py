"""
llm_judge.py — LLM-as-judge (G-Eval) hallucination scorer.
Uses GPT-4o to evaluate whether a response contains hallucinations.
"""

import json
from loguru import logger
from typing import Optional
from src.utils.api_client import query_llm_sync


JUDGE_SYSTEM_PROMPT = """You are an expert factuality judge for AI responses.

Given a question, an AI-generated answer, and optionally a reference answer or context,
evaluate whether the AI answer contains hallucinations.

Respond ONLY with a JSON object (no markdown, no extra text):
{
  "is_hallucinated": true or false,
  "hallucination_score": <float 0.0 to 1.0>,
  "hallucination_type": "<factual|faithfulness|intrinsic|none_detected|unknown>",
  "explanation": "<one sentence summary>",
  "contradictions": ["<specific false claim 1>", ...]
}

Where:
- 0.0 = fully accurate
- 1.0 = heavily hallucinated
"""


def judge_response(
    prompt: str,
    response: str,
    reference: Optional[str] = None,
    context: Optional[str] = None,
    model: str = "gpt-4o",
) -> dict:
    """
    Use an LLM to judge whether a response is hallucinated.

    Args:
        prompt: Original question asked.
        response: LLM response to evaluate.
        reference: Optional ground-truth answer.
        context: Optional retrieved context (for faithfulness check).
        model: LLM to use as judge.

    Returns:
        dict with is_hallucinated, hallucination_score, type, explanation, contradictions.
    """
    user_content = f"Question: {prompt}\n\nAI Answer: {response}"
    if reference:
        user_content += f"\n\nReference Answer: {reference}"
    if context:
        user_content = f"Context:\n{context}\n\n" + user_content

    try:
        raw = query_llm_sync(
            prompt=user_content,
            model=model,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=400,
        )
        result = json.loads(raw)
        logger.info(
            f"[llm_judge] hallucinated={result.get('is_hallucinated')} | "
            f"score={result.get('hallucination_score')}"
        )
        return result
    except json.JSONDecodeError:
        logger.warning("[llm_judge] Response was not valid JSON, using fallback.")
        return {
            "is_hallucinated": False,
            "hallucination_score": 0.0,
            "hallucination_type": "unknown",
            "explanation": "Judge response was not parseable.",
            "contradictions": [],
        }
    except Exception as e:
        logger.error(f"[llm_judge] Judge call failed: {e}")
        return {
            "is_hallucinated": False,
            "hallucination_score": 0.0,
            "hallucination_type": "unknown",
            "explanation": str(e),
            "contradictions": [],
        }


def batch_judge(
    samples: list[dict],
    model: str = "gpt-4o",
    prompt_key: str = "prompt",
    response_key: str = "response",
    reference_key: str = "ground_truth",
) -> list[dict]:
    """
    Run LLM-as-judge over a batch of samples.

    Args:
        samples: List of dicts with prompt, response, optional ground_truth.
        model: Judge model to use.

    Returns:
        Original samples with added 'judge_result' key.
    """
    results = []
    for i, sample in enumerate(samples):
        logger.info(f"[batch_judge] [{i+1}/{len(samples)}]")
        judgment = judge_response(
            prompt=sample.get(prompt_key, ""),
            response=sample.get(response_key, ""),
            reference=sample.get(reference_key),
            model=model,
        )
        results.append({**sample, "judge_result": judgment})

    logger.success(f"[batch_judge] Completed {len(results)} judgments.")
    return results
