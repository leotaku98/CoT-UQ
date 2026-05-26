import json
import logging
import os
from tqdm import tqdm

from config import args
from src.model.llama2_predict import predict, model_init
from src.format.get_cot_prompt import get_cot_prompt
from src.format.get_step_exact_tokens import get_step_exact_tokens

from utils import load_data, print_exp, setup_log, is_effectively_empty, step_exacts_2_list, parse_response_to_dict


def _load_processed_ids(output_path: str) -> set:
    """Return the set of IDs already written to output or error files."""
    processed = set()
    for path in [
        f"{output_path}/output_v1.json",
        f"{output_path}/error_questions/output_v1.json",
    ]:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    processed.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return processed


def llama_inference_refining():
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log = setup_log(args)

    if args.dataset in ["hotpotQA", "2WikimhQA"]:
        question, answer, ids, types = load_data(args)
    else:
        question, answer, ids = load_data(args)

    processed_ids = _load_processed_ids(args.output_path)
    if processed_ids:
        log.info(f"Resuming: skipping {len(processed_ids)} already-processed questions.")

    model, tokenizer, device = model_init(args)
    model.eval()

    for idx, q in enumerate(tqdm(question, total=len(question))):
        if args.dataset in ["gsm8k", "svamp", "ASDiv"]:
            a = answer[idx]
            id = ids[idx]
        else:
            a = answer[idx]
            id = ids[idx]
            t = types[idx]

        if id in processed_ids:
            log.debug(f"Skipping already-processed question id={id}")
            continue

        with open(f"{args.output_path}/output_v1.json", "a", encoding="utf-8") as f:
            log.debug(f"##### This is the --{idx + 1}th-- Question #####")

            cot_prompt = get_cot_prompt(args, q)

            inputs = tokenizer(cot_prompt, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            try_time = 0
            while try_time < args.try_times:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_length_cot,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )

                generated_ids = outputs[0][len(inputs["input_ids"][0]):-1]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                llm_answer, steps_dict, response = parse_response_to_dict(response)

                if generated_ids.size(0) >= args.max_length_cot:
                    log.debug(f'New Reasoning Tokens Are Too Much, Current try is {try_time + 1}')
                    try_time += 1
                elif generated_ids.size(0) == 0:
                    log.debug(f'New Reasoning Tokens Are Null, Current try is {try_time + 1}')
                    try_time += 1
                elif llm_answer is None or llm_answer in ['', ' ']:
                    log.debug(f'New Reasoning Tokens Are None, Current try is {try_time + 1}')
                    try_time += 1
                else:
                    exacts_prompt = get_step_exact_tokens(args, q, response)
                    exact_response = predict(args, exacts_prompt, model, tokenizer)

                    if "NO ANSWER" in exact_response:
                        log.debug(f'Exact Tokens Have NO ANSWER, Current try is {try_time + 1}')
                        try_time += 1
                        continue
                    if not step_exacts_2_list(exact_response):
                        log.debug(f'Exact Tokens Have no contribution scores, Current try is {try_time + 1}')
                        try_time += 1
                        continue

                    exact_response, keywords_list, contributions_list = step_exacts_2_list(exact_response)
                    if len(keywords_list) == 0:
                        log.debug(f'Cannot Extract Effective Keywords, Current try is {try_time + 1}')
                        try_time += 1
                        continue

                    if len(steps_dict) > len(keywords_list):
                        log.debug(
                            f'Len of keywords list doesn\'t match the len of step dict, '
                            f'Current try is {try_time + 1}'
                        )
                        try_time += 1
                        continue

                    keywords_contributions = {}
                    for step_idx, (step_name, _) in enumerate(steps_dict.items()):
                        keywords = keywords_list[step_idx]
                        contributions = contributions_list[step_idx]
                        step_contributions = {}
                        for keyword_idx, keyword in enumerate(keywords):
                            if keyword == 'NO ANSWER':
                                continue
                            step_contributions[keyword] = int(contributions[keyword_idx])
                        keywords_contributions[step_name] = step_contributions

                    if is_effectively_empty(keywords_contributions):
                        log.debug(f'Contributions from All Steps are All None, Current try is {try_time + 1}')
                        try_time += 1
                        continue

                    if args.dataset in ["gsm8k", "svamp", "ASDiv"]:
                        formatted_data = {
                            "id": id,
                            "question": q,
                            "correct answer": a,
                            "llm response": response,
                            "llm answer": llm_answer,
                            "step-wise keywords": exact_response,
                            "keyword contribution": keywords_contributions,
                        }
                    else:
                        formatted_data = {
                            "id": id,
                            "question": q,
                            "correct answer": a,
                            "type": t,
                            "llm response": response,
                            "llm answer": llm_answer,
                            "step-wise keywords": exact_response,
                            "keyword contribution": keywords_contributions,
                        }
                    f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")
                    break

        if try_time >= args.try_times:
            log.debug(
                f'#####The Following Question:#####\n{q}\n'
                f'Has no Meaningful Answer & Explanations, Record and Skip'
            )
            error_dir = f"{args.output_path}/error_questions"
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            with open(f"{args.output_path}/error_questions/output_v1.json", "a", encoding="utf-8") as f:
                if args.dataset in ["gsm8k", "svamp", "ASDiv"]:
                    formatted_data = {
                        "id": id,
                        "question": q,
                        "correct answer": a,
                        "llm response": response,
                        "llm answer": llm_answer,
                    }
                else:
                    formatted_data = {
                        "id": id,
                        "question": q,
                        "correct answer": a,
                        "type": t,
                        "llm response": response,
                        "llm answer": llm_answer,
                    }
                f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    print_exp(args)

    if args.model_engine in ["llama3-1_8B", "llama2-13b"]:
        llama_inference_refining()
    else:
        raise ValueError(f"Invalid model engine: {args.model_engine}")
