"""
prompt_engineering.py — Prompt engineering mitigation techniques.
Implements zero-shot, few-shot, and chain-of-thought prompting strategies.
"""

from loguru import logger
from src.utils.api_client import query_llm_sync

# ── System Prompts ────────────────────────────────────────────────────────────
BASE_PROMPT = (
    "You are a factual AI assistant. Answer questions accurately and concisely. "
    "If you are not certain about a fact, say so explicitly."
)

FEW_SHOT_PROMPT = (
    "You are a factual AI assistant. Answer questions accurately using the examples below.\n\n"
    "Example 1:\nQ: What year did World War II end?\nA: World War II ended in 1945.\n\n"
    "Example 2:\nQ: Who invented the telephone?\nA: Alexander Graham Bell is credited with "
    "inventing the telephone in 1876.\n\n"
    "Now answer the following question factually:"
)

COT_PROMPT = (
    "You are a factual AI assistant. Before answering, think step-by-step. "
    "Show your reasoning process, then provide a concise final answer. "
    "If any step involves a fact you're unsure about, flag it explicitly.\n\n"
    "Format:\nThinking: <step-by-step reasoning>\nAnswer: <final concise answer>"
)

FACTUALITY_REINFORCED_PROMPT = (
    "You are a factual AI assistant with the following rules:\n"
    "1. ONLY state facts you are highly confident about.\n"
    "2. If uncertain, explicitly say 'I'm not certain, but...' or 'I don't know.'\n"
    "3. NEVER guess or fabricate information.\n"
    "4. Be concise and accurate."
)


def zero_shot(prompt: str, model: str = "gpt-4o") -> str:
    """Standard zero-shot prompting with factuality-focused system prompt."""
    return query_llm_sync(prompt=prompt, model=model, system_prompt=BASE_PROMPT)


def few_shot(prompt: str, model: str = "gpt-4o") -> str:
    """Few-shot prompting with factual examples in the system prompt."""
    return query_llm_sync(prompt=prompt, model=model, system_prompt=FEW_SHOT_PROMPT)


def chain_of_thought(prompt: str, model: str = "gpt-4o") -> str:
    """Chain-of-thought: forces model to reason step-by-step before answering."""
    return query_llm_sync(prompt=prompt, model=model, system_prompt=COT_PROMPT)


def factuality_reinforced(prompt: str, model: str = "gpt-4o") -> str:
    """Strict factuality-reinforced prompt that discourages any guessing."""
    return query_llm_sync(
        prompt=prompt, model=model, system_prompt=FACTUALITY_REINFORCED_PROMPT
    )


def apply_technique(prompt: str, technique: str, model: str = "gpt-4o") -> dict:
    """
    Apply a named prompting technique to a question.

    Args:
        prompt: The question to answer.
        technique: One of 'zero_shot', 'few_shot', 'chain_of_thought', 'factuality_reinforced'.
        model: LLM to use.

    Returns:
        Dict with 'answer', 'technique', 'model'.
    """
    dispatch = {
        "zero_shot":             zero_shot,
        "few_shot":              few_shot,
        "chain_of_thought":      chain_of_thought,
        "factuality_reinforced": factuality_reinforced,
    }

    fn = dispatch.get(technique, zero_shot)
    logger.info(f"[prompt_engineering] technique={technique} | model={model}")
    answer = fn(prompt, model)
    return {"answer": answer, "technique": technique, "model": model}
