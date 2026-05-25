# -*- coding: utf-8 -*-
"""Evaluate self-probing UQ: label samples and compute AUROC + F1 for all 5 variants."""

import json
import logging
import os
import time
from typing import Optional

import torch
from tqdm import tqdm

from config import args
from utils import print_exp
from torchmetrics import AUROC, F1Score

from openai import OpenAI, APIError

SYSTEM_PROMPT_ORACLE_EQUIVALENCY = (
    "You are an automated grading assistant helping a teacher grade student answers."
)

PROMPT_ANSWER_KEY_EQUIVALENCY = (
    "The problem is: <question>\n\n The correct answer for this problem is: <ground-truth>\n "
    + "A student submitted the answer: <prediction>\n "
    + "The student's answer must be correct and specific but not overcomplete "
    + "(for example, if they provide two different answers, they did not get the question right). "
    + "However, small differences in formatting should not be penalized (for example, 'New York City' is equivalent to 'NYC'). "
    + "Did the student provide an equivalent answer to the ground truth? Please answer yes or no without any explanation: "
)

VARIANTS = ["baseline", "allkeyword", "keykeyword", "allstep", "keystep"]


def _model_output_path() -> str:
    """Infer per-model output path from model_id and dataset."""
    model_name = args.model_id.split("/")[-1]
    return f"output/{model_name}/{args.dataset}"


def openai_query(system_prompt, prompt, openai_model_name="gpt-4o-mini"):
    """Query OpenAI with retry on API errors."""
    client = OpenAI()
    sampled_response = None
    while sampled_response is None:
        try:
            response = client.chat.completions.create(
                model=openai_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            sampled_response = response.choices[0].message.content
        except APIError:
            logging.exception("OpenAI API Error.", exc_info=True)
            time.sleep(1)
    return sampled_response


def label_samples(output_path):
    """Write output_v1_w_labels.json with correctness labels for each question."""
    with open(f"{output_path}/output_v1.json", "r", encoding="utf-8") as f:
        json_data = [json.loads(line) for line in f if line.strip()]

    with open(f"{output_path}/output_v1_w_labels.json", "a", encoding="utf-8") as f:
        for line in tqdm(json_data, total=len(json_data)):
            question = line["question"]
            correct_answer = line["correct answer"]
            llm_answer = line["llm answer"]

            if args.dataset in ["gsm8k", "svamp", "ASDiv"]:
                label = (
                    str(correct_answer) in llm_answer.lower()
                    or str(int(correct_answer)) in llm_answer.lower()
                )
            else:
                prompt = (
                    PROMPT_ANSWER_KEY_EQUIVALENCY
                    .replace("<ground-truth>", str(correct_answer))
                    .replace("<prediction>", llm_answer)
                    .replace("<question>", question)
                )
                sampled_response = openai_query(SYSTEM_PROMPT_ORACLE_EQUIVALENCY, prompt)
                label = "yes" in sampled_response.strip().lower()

            f.write(json.dumps({
                "id": line["id"],
                "question": question,
                "correct answer": correct_answer,
                "llm answer": llm_answer,
                "label": label,
                "llm response": line["llm response"],
                "step-wise keywords": line["step-wise keywords"],
                "keyword contribution": line["keyword contribution"],
            }, ensure_ascii=False) + "\n")


def compute_metrics(output_path, variant):
    # type: (str, str) -> Optional[dict]
    """Compute AUROC and F1 for one variant. Returns None if output file missing."""
    labels_path = f"{output_path}/output_v1_w_labels.json"
    confidence_path = f"{output_path}/confidences/output_v1_self-probing-{variant}.json"

    if not os.path.exists(confidence_path):
        return None

    label_dict = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                label_dict[entry["question"]] = 1 if entry["label"] else 0

    confidences, targets = [], []
    with open(confidence_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                confidences.append(entry["confidence"])
                targets.append(label_dict[entry["question"]])

    conf_t = torch.tensor(confidences)
    tgt_t = torch.tensor(targets)

    auroc_value = float(AUROC(task="binary")(conf_t, tgt_t))
    f1_value = float(F1Score(task="binary")(conf_t, tgt_t))

    return {"auroc": round(auroc_value, 4), "f1": round(f1_value, 4)}


if __name__ == "__main__":
    print_exp(args)

    output_path = _model_output_path()
    model_name = args.model_id.split("/")[-1]

    print(f"\nLabelling samples in {output_path} ...")
    label_samples(output_path)

    print(f"\n=== Results: {model_name} / {args.dataset} ===")
    variant_results = {}
    for variant in VARIANTS:
        metrics = compute_metrics(output_path, variant)
        if metrics is None:
            print(f"  [{variant}] (no output file, skipping)")
        else:
            print(f"  [{variant}]  AUROC: {metrics['auroc']:.4f}  F1: {metrics['f1']:.4f}")
            variant_results[variant] = metrics

    result_dir = f"output/result/{args.dataset}"
    os.makedirs(result_dir, exist_ok=True)
    result_path = f"{result_dir}/{model_name}.json"
    existing = json.load(open(result_path)) if os.path.exists(result_path) else {}
    existing[model_name] = variant_results
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {result_path}")
