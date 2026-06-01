import json
import os
from tqdm import tqdm

from config import args
from src.model.llama2_predict import model_init
from src.format.get_cot_prompt import get_cot_prompt
from utils import load_data, print_exp, setup_log, parse_response_to_dict

NUM_SAMPLES = 10


def _load_processed_ids(output_path: str) -> set:
    """Return the set of IDs already written to ensemble_v1.json."""
    processed = set()
    path = f"{output_path}/ensemble_v1.json"
    if not os.path.exists(path):
        return processed
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


def sampling_inference():
    os.makedirs(args.output_path, exist_ok=True)

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
        id = ids[idx]
        a = answer[idx]

        if id in processed_ids:
            log.debug(f"Skipping already-processed question id={id}")
            continue

        cot_prompt = get_cot_prompt(args, q)
        inputs = tokenizer(cot_prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        samples = []
        for _ in range(NUM_SAMPLES):
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_length_cot,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_ids = outputs[0][len(inputs["input_ids"][0]):-1]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            llm_answer, _, response = parse_response_to_dict(response)
            samples.append({
                "llm response": response,
                "llm answer": llm_answer,
            })

        formatted_data = {
            "id": id,
            "question": q,
            "correct answer": a,
            "samples": samples,
        }

        with open(f"{args.output_path}/ensemble_v1.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print_exp(args)

    if args.model_engine in ["llama3-1_8B", "llama2-13b"]:
        sampling_inference()
    else:
        raise ValueError(f"Invalid model engine: {args.model_engine}")
