# -*- coding: utf-8 -*-
"""Self-probing uncertainty quantification with 5 context variants."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from tqdm import tqdm

from config import args
from src.model.api_client import chat_complete
from utils import (
    extract_keywords, extract_keykeywords, extract_keystep,
    extract_probing_confidence,
)

VARIANTS = ["baseline", "allkeyword", "keykeyword", "allstep", "keystep"]

_PROBING_SUFFIX = (
    "Q: How likely is the above answer to be correct? Please first show your reasoning "
    "concisely and then answer with the following format:\n"
    "```Confidence: [the probability of answer {answer} to be correct, not the one you "
    "think correct, please only include the numerical number]%```\n"
    "Confidence: "
)


def build_self_probing_prompt(
    question: str,
    llm_answer: str,
    variant: str,
    keyword_contribution: dict,
    llm_response: str,
) -> str:
    """Build a self-probing prompt for a given context variant.

    Args:
        question: The original question.
        llm_answer: The model's answer to evaluate.
        variant: One of 'baseline', 'allkeyword', 'keykeyword', 'allstep', 'keystep'.
        keyword_contribution: Dict of {step: {keyword: score}} from inference_refining.
        llm_response: Full step-by-step reasoning text.

    Raises:
        ValueError: If variant is not one of the five supported values.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant!r}. Must be one of {VARIANTS}.")

    base = f"Question: {question}\nPossible Answer: {llm_answer}\n"

    if variant == "baseline":
        context = ""
    elif variant == "allkeyword":
        keywords = extract_keywords(keyword_contribution)
        context = f"Keywords during reasoning to the possible answer: {keywords}\n"
    elif variant == "keykeyword":
        keywords = extract_keykeywords(keyword_contribution)
        context = f"The most important keywords during reasoning: {keywords}\n"
    elif variant == "allstep":
        context = f"A step-by-step reasoning to the possible answer: {llm_response}\n"
    elif variant == "keystep":
        key_step = extract_keystep(llm_response, keyword_contribution)
        context = f"The most critical step in reasoning to the possible answer: {key_step}\n"

    suffix = _PROBING_SUFFIX.format(answer=llm_answer)
    return base + context + suffix


def _probe_question(task: tuple) -> dict | None:
    """Run self-probing for one question under one variant.

    Returns a result dict on success, or None if all retries are exhausted.
    """
    line, variant, gen_kwargs, model_id, provider, try_times = task

    question = line["question"]
    llm_answer = line.get("llm answer", "")
    llm_response = line.get("llm response", "")
    keyword_contribution = line.get("keyword contribution", {})

    if not llm_answer or not keyword_contribution:
        return None

    prompt = build_self_probing_prompt(
        question, llm_answer, variant, keyword_contribution, llm_response
    )

    for try_time in range(try_times):
        response = chat_complete(prompt, model_id, provider, **gen_kwargs)
        confidence = extract_probing_confidence(response)

        if confidence is None:
            print(f"  [{variant}] Cannot extract confidence (try {try_time + 1}): {response[:80]}")
            continue

        return {
            "question": question,
            "correct answer": line["correct answer"],
            "llm answer": llm_answer,
            "confidence": confidence,
            "probing response": response,
        }

    print(f"  [{variant}] Skipping after {try_times} failed tries")
    return None


def _load_processed_questions(out_path: str) -> set:
    """Return the set of question strings already written to a confidence file."""
    if not os.path.exists(out_path):
        return set()
    processed = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    processed.add(json.loads(line)["question"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return processed


def self_probing_uncertainty() -> None:
    """Run self-probing for all 5 variants over output_v1.json."""
    with open(f"{args.output_path}/output_v1.json", "r", encoding="utf-8") as f:
        json_data = [json.loads(line) for line in f if line.strip()]

    gen_kwargs = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
    }

    output_dir = f"{args.output_path}/confidences/"
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    variant_timings = {}

    for variant in VARIANTS:
        out_path = f"{output_dir}output_v1_self-probing-{variant}.json"
        print(f"\n=== Variant: {variant} → {out_path} ===")

        if args.resume:
            processed_questions = _load_processed_questions(out_path)
            variant_data = [line for line in json_data if line["question"] not in processed_questions]
            print(f"  Resuming: {len(processed_questions)} already done, {len(variant_data)} remaining.")
        else:
            if os.path.exists(out_path):
                open(out_path, "w").close()
            variant_data = json_data

        tasks = [
            (line, variant, gen_kwargs, args.model_id, args.provider, args.try_times)
            for line in variant_data
        ]

        v_start = time.time()
        success_count = 0

        with open(out_path, "a", encoding="utf-8") as f_out:
            with ThreadPoolExecutor(max_workers=2) as executor:
                for result in tqdm(executor.map(_probe_question, tasks), total=len(tasks), desc=variant):
                    if result is not None:
                        success_count += 1
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

        v_elapsed = time.time() - v_start
        variant_timings[variant] = {
            "elapsed_seconds": round(v_elapsed, 1),
            "questions_processed": len(tasks),
            "success": success_count,
        }

    total_elapsed = time.time() - start_time
    elapsed_human = f"{int(total_elapsed // 3600)}h {int(total_elapsed % 3600 // 60)}m {int(total_elapsed % 60)}s"
    timing_path = f"{args.output_path}/timing.json"
    timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
    timing["stepuq"] = {
        "model": args.model_id,
        "dataset": args.dataset,
        "started_at": datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(total_elapsed, 1),
        "elapsed_human": elapsed_human,
        "variants": variant_timings,
    }
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(f"\nDone in {elapsed_human} — timing saved to {timing_path}")


if __name__ == "__main__":
    self_probing_uncertainty()
