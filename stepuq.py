# -*- coding: utf-8 -*-
"""Self-probing uncertainty quantification with 5 context variants."""

import json
import os
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

    for variant in VARIANTS:
        out_path = f"{output_dir}output_v1_self-probing-{variant}.json"
        print(f"\n=== Variant: {variant} → {out_path} ===")

        with open(out_path, "a", encoding="utf-8") as f_out:
            for line in tqdm(json_data, desc=variant):
                question = line["question"]
                llm_answer = line.get("llm answer", "")
                llm_response = line.get("llm response", "")
                keyword_contribution = line.get("keyword contribution", {})

                if not llm_answer or not keyword_contribution:
                    continue

                prompt = build_self_probing_prompt(
                    question, llm_answer, variant, keyword_contribution, llm_response
                )

                try_time = 0
                while try_time < args.try_times:
                    response = chat_complete(
                        prompt, args.model_id, args.provider, **gen_kwargs
                    )
                    confidence = extract_probing_confidence(response)

                    if confidence is None:
                        print(f"  Cannot extract confidence (try {try_time + 1}): {response[:80]}")
                        try_time += 1
                        continue

                    f_out.write(json.dumps({
                        "question": question,
                        "correct answer": line["correct answer"],
                        "llm answer": llm_answer,
                        "confidence": confidence,
                        "probing response": response,
                    }, ensure_ascii=False) + "\n")
                    break

                if try_time >= args.try_times:
                    print(f"  Skipping after {args.try_times} failed tries")


if __name__ == "__main__":
    self_probing_uncertainty()
