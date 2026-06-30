"""
pipeline.py — Full evaluation pipeline runner.
Orchestrates: load data → call LLM → judge → score metrics → save results.
"""

import pandas as pd
from pathlib import Path
from loguru import logger
from typing import List, Optional

from src.utils.data_loader import load_jsonl, save_evaluation_outputs
from src.analysis.output_profiler import profile_outputs
from src.evaluation.llm_judge import batch_judge
from src.evaluation.metrics import score_sample


def run_evaluation_pipeline(
    prompts: List[str],
    ground_truths: Optional[List[str]] = None,
    model: str = "gpt-4o",
    mitigation: str = "none",
    judge_model: str = "gpt-4o",
    output_filename: str = "pipeline_results.jsonl",
    concurrency: int = 3,
) -> pd.DataFrame:
    """
    Run the complete hallucination evaluation pipeline end-to-end.

    Steps:
    1. Profile outputs: call LLM for each prompt.
    2. LLM judge: score each response for hallucinations.
    3. Metrics: compute BERTScore, ROUGE-L, semantic similarity.
    4. Save and return results as a DataFrame.

    Args:
        prompts: Questions to ask the LLM.
        ground_truths: Reference answers (same length as prompts).
        model: LLM to evaluate.
        mitigation: Mitigation technique name (metadata only).
        judge_model: LLM to use as evaluator/judge.
        output_filename: Filename to save results in data/evaluation/.
        concurrency: Parallel LLM calls for profiling.

    Returns:
        pandas DataFrame with all scores.
    """
    logger.info(
        f"Starting evaluation pipeline | model={model} | "
        f"mitigation={mitigation} | n={len(prompts)}"
    )

    # Step 1: Generate LLM outputs
    outputs = profile_outputs(
        prompts=prompts,
        model=model,
        mitigation=mitigation,
        ground_truths=ground_truths,
        save_filename=f"raw_{output_filename}",
        concurrency=concurrency,
    )

    # Step 2: LLM judge scoring
    judged = batch_judge(outputs, model=judge_model)

    # Step 3: Compute metric scores
    final_results = []
    for sample in judged:
        metrics = score_sample(
            candidate=sample.get("response", ""),
            reference=sample.get("ground_truth"),
        )
        final_results.append({
            **sample,
            **metrics,
            "hallucination_score": sample["judge_result"].get("hallucination_score", 0.0),
            "is_hallucinated":     sample["judge_result"].get("is_hallucinated", False),
            "hallucination_type":  sample["judge_result"].get("hallucination_type", "unknown"),
            "explanation":         sample["judge_result"].get("explanation", ""),
        })

    # Step 4: Save
    save_evaluation_outputs(final_results, filename=output_filename)

    df = pd.DataFrame(final_results)
    hallucination_rate = df["is_hallucinated"].mean() * 100
    logger.success(
        f"Pipeline complete | hallucination_rate={hallucination_rate:.1f}% | "
        f"saved to data/evaluation/{output_filename}"
    )

    return df


def compare_techniques(results_files: dict[str, str]) -> pd.DataFrame:
    """
    Compare hallucination results across multiple technique runs.

    Args:
        results_files: Dict mapping technique name to JSONL filename.
                       e.g. {"none": "none_results.jsonl", "rag": "rag_results.jsonl"}

    Returns:
        DataFrame with one row per technique showing aggregate metrics.
    """
    rows = []
    for technique, filename in results_files.items():
        records = load_jsonl(
            Path(__file__).resolve().parents[2] / "data" / "evaluation" / filename
        )
        if not records:
            logger.warning(f"No records for technique '{technique}'")
            continue

        df = pd.DataFrame(records)
        rows.append({
            "technique":           technique,
            "n_samples":           len(df),
            "hallucination_rate":  round(df["is_hallucinated"].mean() * 100, 2),
            "avg_hallucination_score": round(df["hallucination_score"].mean(), 4),
            "avg_bert_score":      round(df["bert_score"].dropna().mean(), 4) if "bert_score" in df else None,
            "avg_rouge_l":         round(df["rouge_l"].dropna().mean(), 4) if "rouge_l" in df else None,
        })

    comparison = pd.DataFrame(rows).sort_values("hallucination_rate")
    logger.info(f"\n{comparison.to_string(index=False)}")
    return comparison
