"""
llm_service.py — AI Model Layer
Unified LLM caller using litellm. Handles OpenAI, Gemini, Mistral via one interface.
"""

import litellm
from loguru import logger
from typing import Optional, List
from deployment.api.config import get_settings

settings = get_settings()

# Configure litellm keys
import os
if settings.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
if settings.GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
if settings.GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY


# ── System prompts ────────────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = (
    "You are a factual assistant. Answer questions accurately and concisely. "
    "If you are not certain about a fact, say so explicitly rather than guessing."
)

FEW_SHOT_SYSTEM_PROMPT = (
    "You are a factual assistant. Answer questions accurately using the examples below as a guide.\n\n"
    "Example 1:\nQ: What year did World War II end?\nA: World War II ended in 1945.\n\n"
    "Example 2:\nQ: Who invented the telephone?\nA: Alexander Graham Bell is credited with inventing the telephone in 1876.\n\n"
    "Now answer the following question factually and concisely."
)

COT_SYSTEM_PROMPT = (
    "You are a factual assistant. Before answering, think step-by-step about what you know. "
    "Show your reasoning, then provide a final concise answer. "
    "If a step involves a fact you are unsure about, flag it explicitly."
)

FACTUALITY_JUDGE_PROMPT = """You are an expert factuality judge. Given a question and an AI-generated answer, 
determine if the answer contains hallucinations (false or unsupported claims).

Respond with a JSON object containing:
- "is_hallucinated": true or false
- "hallucination_type": one of ["factual", "faithfulness", "intrinsic", "none_detected"]
- "hallucination_score": float between 0.0 (fully faithful) and 1.0 (heavily hallucinated)
- "explanation": brief explanation of your judgment
- "contradictions": list of specific false claims detected (empty list if none)

Be strict and accurate."""


def _get_system_prompt(mitigation: str) -> str:
    """Select system prompt based on mitigation technique."""
    mapping = {
        "few_shot": FEW_SHOT_SYSTEM_PROMPT,
        "chain_of_thought": COT_SYSTEM_PROMPT,
    }
    return mapping.get(mitigation, BASE_SYSTEM_PROMPT)


async def call_llm(
    prompt: str,
    model: str = None,
    mitigation: str = "none",
    context: Optional[str] = None,
    temperature: float = None,
    max_tokens: int = None,
) -> str:
    """
    Call any LLM via litellm.
    Automatically injects the appropriate system prompt based on mitigation technique.
    If context is provided (e.g., from RAG), it's prepended to the user message.
    """
    model = model or settings.DEFAULT_MODEL
    temperature = temperature if temperature is not None else settings.TEMPERATURE
    max_tokens = max_tokens or settings.MAX_TOKENS

    system_prompt = _get_system_prompt(mitigation)

    # Build user message
    user_content = prompt
    if context:
        user_content = (
            f"Use the following context to answer the question accurately:\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {prompt}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    logger.info(f"Calling LLM | model={model} | mitigation={mitigation} | prompt_len={len(prompt)}")

    response = await litellm.acompletion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    result = response.choices[0].message.content.strip()
    logger.debug(f"LLM response (first 120 chars): {result[:120]}")
    return result


async def call_llm_judge(
    prompt: str,
    response: str,
    context: Optional[str] = None,
    model: str = None,
) -> dict:
    """
    Use GPT-4o as a factuality judge (LLM-as-judge / G-Eval pattern).
    Returns a dict with hallucination score, type, and explanation.
    """
    import json

    judge_user = (
        f"Question: {prompt}\n\n"
        f"AI Answer: {response}\n"
    )
    if context:
        judge_user = f"Context provided to AI:\n{context}\n\n" + judge_user

    messages = [
        {"role": "system", "content": FACTUALITY_JUDGE_PROMPT},
        {"role": "user", "content": judge_user},
    ]

    model = model or settings.DEFAULT_MODEL
    logger.info(f"Running LLM judge | model={model}")

    llm_response = await litellm.acompletion(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    raw = llm_response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM judge returned non-JSON, using fallback.")
        result = {
            "is_hallucinated": False,
            "hallucination_type": "unknown",
            "hallucination_score": 0.5,
            "explanation": raw,
            "contradictions": [],
        }

    return result


async def call_llm_n_times(
    prompt: str,
    model: str,
    n: int = 5,
    temperature: float = 0.7,
) -> List[str]:
    """
    Call the LLM N times with higher temperature for self-consistency voting.
    Returns a list of N responses.
    """
    import asyncio

    tasks = [
        call_llm(prompt, model=model, temperature=temperature)
        for _ in range(n)
    ]
    results = await asyncio.gather(*tasks)
    logger.info(f"Self-consistency: generated {n} responses for voting")
    return list(results)
