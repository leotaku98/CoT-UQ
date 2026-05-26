import os
import json
from config import args
from tqdm import tqdm

from utils import extract_keystep, extract_probing_confidence
from src.model.llama2_predict import predict, model_init


def self_probing_uncertainty():
    with open(f"{args.output_path}/output_v1.json", 'r', encoding='utf-8') as f:
        json_data = [json.loads(line) for line in f.readlines()]

    model, tokenizer, device = model_init(args)

    output_dir = f"{args.output_path}/confidences/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, line in enumerate(tqdm(json_data, total=len(json_data))):
        with open(f"{args.output_path}/confidences/output_v1_{args.uq_engine}.json", "a", encoding="utf-8") as f:
            question = line['question']
            correct_answer = line['correct answer']
            llm_answer = line['llm answer']
            llm_response = line['llm response']
            contribution_scores = line['keyword contribution']

            if not llm_answer or not llm_response or not contribution_scores:
                continue

            key_step = extract_keystep(llm_response, contribution_scores)

            prompt = (
                f"Question: {question}\n"
                f"Possible Answer: {llm_answer}\n"
                f"The most critical step in reasoning to the possible answer: {key_step}\n"
                f"Q: Considering this critical reasoning step as additional information, how likely is the above answer to be correct? "
                f"Please first show your reasoning concisely and then answer with the following format:\n"
                f"```Confidence: [the probability of answer {llm_answer} to be correct, "
                f"not the one you think correct, please only include the numerical number]%```\n"
                f"Confidence: "
            )

            try_time = 0
            while try_time < args.try_times:
                response = predict(args, prompt, model, tokenizer)
                confidence = extract_probing_confidence(response)

                if confidence is None:
                    print(f"Cannot extract confidence, Please check the response: {response}")
                    print(f"Current try is {try_time + 1}")
                    try_time += 1
                    continue

                formatted_data = {
                    "question": question,
                    "correct answer": correct_answer,
                    "llm answer": llm_answer,
                    "confidence": confidence,
                    "probing response": response,
                }
                f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")
                break

        if try_time >= args.try_times:
            print(f'#####Cannot extract confidence from:#####\n{response}\nRecord and Skip')
            error_dir = f"{args.output_path}/confidences/probing_errors"
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            with open(f"{args.output_path}/confidences/probing_errors/output_v1_self_probing_keystep.json",
                      "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    self_probing_uncertainty()
