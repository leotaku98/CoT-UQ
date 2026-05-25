# -*- coding: utf-8 -*-
"""CoT generation and keyword/contribution extraction via black-box API."""

import json
import os
from tqdm import tqdm

from config import args
from src.model.api_client import chat_complete
from src.format.get_cot_prompt import get_cot_prompt
from src.format.get_step_exact_tokens import get_step_exact_tokens
from utils import (
    load_data, print_exp, setup_log, is_effectively_empty,
    step_exacts_2_list, parse_response_to_dict,
)


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

    for idx, q in enumerate(tqdm(question, total=len(question))):
        a = answer[idx]
        id_ = ids[idx]
        t = types[idx]

        log.debug(f"##### Question {idx + 1} #####")
        cot_prompt = get_cot_prompt(args, q)

        try_time = 0
        llm_answer = None
        response = None

        while try_time < args.try_times:
            response_text = chat_complete(cot_prompt, args.model_id, args.provider, **gen_kwargs)
            llm_answer, steps_dict, response = parse_response_to_dict(response_text)

            if llm_answer is None or llm_answer in ["", " "]:
                log.debug(f"No valid Final Answer, try {try_time + 1}")
                try_time += 1
                continue

            if not steps_dict:
                log.debug(f"No reasoning steps parsed, try {try_time + 1}")
                try_time += 1
                continue

            exacts_prompt = get_step_exact_tokens(args, q, response)
            exact_text = chat_complete(exacts_prompt, args.model_id, args.provider, **gen_kwargs)

            parsed = step_exacts_2_list(exact_text)
            if not parsed:
                log.debug(f"Keyword parsing failed, try {try_time + 1}")
                try_time += 1
                continue

            exact_response, keywords_list, contributions_list = parsed

            if len(keywords_list) == 0:
                log.debug(f"No keywords extracted, try {try_time + 1}")
                try_time += 1
                continue

            if len(steps_dict) > len(keywords_list):
                log.debug(f"Steps/keywords count mismatch, try {try_time + 1}")
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
                log.debug(f"All keyword contributions empty, try {try_time + 1}")
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

            with open(f"{args.output_path}/output_v1.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")
            break

        if try_time >= args.try_times:
            log.debug(f"Question {idx + 1} exceeded retry limit — skipping")
            error_dir = f"{args.output_path}/error_questions"
            os.makedirs(error_dir, exist_ok=True)
            error_data = {
                "id": id_,
                "question": q,
                "correct answer": a,
                "llm answer": llm_answer,
            }
            if t is not None:
                error_data["type"] = t
            with open(f"{error_dir}/output_v1.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(error_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print_exp(args)
    api_inference_refining()
