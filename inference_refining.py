# -*- coding: utf-8 -*-
"""CoT generation and keyword/contribution extraction via black-box API."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from config import args
from src.model.api_client import chat_complete
from src.format.get_cot_prompt import get_cot_prompt
from src.format.get_step_exact_tokens import get_step_exact_tokens
from utils import (
    load_data, print_exp, setup_log, is_effectively_empty,
    step_exacts_2_list, parse_response_to_dict,
)


def _process_question(task: tuple) -> dict:
    """Process one question: generate CoT + extract keyword contributions.

    Returns a dict with keys 'type' ('success'|'error') and 'data'.
    """
    idx, q, a, id_, t, gen_kwargs, output_path = task

    cot_prompt = get_cot_prompt(args, q)
    try_time = 0
    llm_answer = None
    response = None

    while try_time < args.try_times:
        response_text = chat_complete(cot_prompt, args.model_id, args.provider, **gen_kwargs)
        llm_answer, steps_dict, response = parse_response_to_dict(response_text)

        if llm_answer is None or llm_answer in ["", " "]:
            try_time += 1
            continue

        if not steps_dict:
            try_time += 1
            continue

        exacts_prompt = get_step_exact_tokens(args, q, response)
        exact_text = chat_complete(exacts_prompt, args.model_id, args.provider, **gen_kwargs)

        parsed = step_exacts_2_list(exact_text)
        if not parsed:
            try_time += 1
            continue

        exact_response, keywords_list, contributions_list = parsed

        if len(keywords_list) == 0 or len(steps_dict) > len(keywords_list):
            try_time += 1
            continue

        keywords_contributions = {}
        for step_idx, (step_name, _) in enumerate(steps_dict.items()):
            keywords = keywords_list[step_idx]
            contributions = contributions_list[step_idx]
            if len(keywords) == 1 and keywords[0] == "NO ANSWER":
                continue
            keywords_contributions[step_name] = {
                kw: int(contributions[ki]) for ki, kw in enumerate(keywords)
            }

        if is_effectively_empty(keywords_contributions):
            try_time += 1
            continue

        formatted_data = {
            "id": id_,
            "question": q,
            "correct answer": a,
            "llm response": response,
            "llm answer": llm_answer,
            "step-wise keywords": exact_response,
            "keyword contribution": keywords_contributions,
        }
        if t is not None:
            formatted_data["type"] = t

        return {"type": "success", "data": formatted_data, "output_path": output_path}

    error_data = {"id": id_, "question": q, "correct answer": a, "llm answer": llm_answer}
    if t is not None:
        error_data["type"] = t
    return {"type": "error", "data": error_data, "output_path": output_path}


def _load_processed_ids(output_path: str) -> set:
    """Return the set of question IDs already written to output_v1.json."""
    out_file = f"{output_path}/output_v1.json"
    if not os.path.exists(out_file):
        return set()
    processed = set()
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    processed.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return processed


def api_inference_refining() -> None:
    """Generate CoT responses and extract step-wise keyword contributions."""
    os.makedirs(args.output_path, exist_ok=True)
    log = setup_log(args)

    if args.dataset in ["hotpotQA", "2WikimhQA"]:
        question, answer, ids, types = load_data(args)
    else:
        question, answer, ids = load_data(args)
        types = [None] * len(question)

    gen_kwargs = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
    }

    out_file = f"{args.output_path}/output_v1.json"
    if args.resume:
        processed_ids = _load_processed_ids(args.output_path)
        tasks = [
            (idx, q, answer[idx], ids[idx], types[idx], gen_kwargs, args.output_path)
            for idx, q in enumerate(question)
            if ids[idx] not in processed_ids
        ]
        print(f"Resuming: {len(processed_ids)} already done, {len(tasks)} remaining.")
    else:
        if os.path.exists(out_file):
            open(out_file, "w").close()
        tasks = [
            (idx, q, answer[idx], ids[idx], types[idx], gen_kwargs, args.output_path)
            for idx, q in enumerate(question)
        ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        for result in tqdm(executor.map(_process_question, tasks), total=len(tasks)):
            if result["type"] == "success":
                with open(f"{result['output_path']}/output_v1.json", "a", encoding="utf-8") as f:
                    f.write(json.dumps(result["data"], ensure_ascii=False) + "\n")
            else:
                error_dir = f"{result['output_path']}/error_questions"
                os.makedirs(error_dir, exist_ok=True)
                log.debug(f"Question exceeded retry limit: {result['data']['question'][:60]}")
                with open(f"{error_dir}/output_v1.json", "a", encoding="utf-8") as f:
                    f.write(json.dumps(result["data"], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print_exp(args)
    api_inference_refining()
