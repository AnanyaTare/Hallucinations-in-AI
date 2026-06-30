"""
api_client.py — Unified LLM API client utility.
Thin wrapper around litellm providing a simple async call interface.
"""

import litellm
from loguru import logger
from typing import Optional, List
from src.utils.logger import setup_logger

setup_logger()


async def query_llm(
    prompt: str,
    model: str = "gpt-4o",
    system_prompt: str = "You are a factual AI assistant.",
    context: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Send a prompt to any LLM via litellm and return the text response.

    Args:
        prompt: The user question or instruction.
        model: LiteLLM model string (e.g. 'gpt-4o', 'gemini/gemini-1.5-pro').
        system_prompt: System-level instruction for the model.
        context: Optional document context to prepend (RAG use case).
        temperature: Sampling temperature (0 = deterministic).
        max_tokens: Maximum tokens in the response.

    Returns:
        The model's response as a plain string.
    """
    user_content = prompt
    if context:
        user_content = (
            f"Use the following context to answer accurately:\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {prompt}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    logger.info(f"[api_client] Querying model={model} | temp={temperature} | prompt_len={len(prompt)}")

    response = await litellm.acompletion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    result = response.choices[0].message.content.strip()
    logger.debug(f"[api_client] Response preview: {result[:100]}")
    return result


def query_llm_sync(
    prompt: str,
    model: str = "gpt-4o",
    system_prompt: str = "You are a factual AI assistant.",
    context: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Synchronous version of query_llm.
    Use in scripts and notebooks where asyncio is not available.
    """
    import asyncio
    return asyncio.run(
        query_llm(prompt, model, system_prompt, context, temperature, max_tokens)
    )
