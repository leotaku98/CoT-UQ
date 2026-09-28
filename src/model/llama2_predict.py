import torch
import time
import os
import json
import re
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
# from peft import PeftModel

HF_NAMES = {
    'llama3-1_8B': 'meta-llama/Llama-3.1-8B',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'qwen2.5-3b': 'Qwen/Qwen2.5-3B-Instruct',
    'qwen3-8B': 'Qwen/Qwen3-8B',
}

def model_init(args):
    model_name = HF_NAMES[args.model_engine]
    device = torch.device("cuda:0")
    if "llama" in args.model_engine:
        if args.model_engine == 'llama2-13b':
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif "qwen" in args.model_engine:
        if args.model_engine == 'qwen3-8B':
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif "mistral" in args.model_engine:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f"Invalid model engine: {args.model_engine}")
    return model, tokenizer, device

class StopOnConfidence(StoppingCriteria):
    """Halt generation as soon as a confidence percentage has been emitted.

    Self-probing only needs the percentage, which the model produces in its first few
    tokens; without this it keeps generating to max_new_tokens and the rest is discarded
    by extract_probing_confidence(), which reads the *first* ``\\d+%`` match.
    """

    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0][self.prompt_len:], skip_special_tokens=True)
        return re.search(r"\d+(\.\d+)?%", text) is not None


def predict(args, prompt, model, tokenizer, stop_on_confidence: bool = False):
    """Greedy-decode a response.

    When stop_on_confidence is set, generation halts at the first percentage rather than
    running to max_new_tokens. Off by default so CoT inference is unaffected.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    prompt_len = len(inputs["input_ids"][0])

    stopping_criteria = None
    if stop_on_confidence:
        stopping_criteria = StoppingCriteriaList([StopOnConfidence(tokenizer, prompt_len)])

    generate_ids = model.generate(
        **inputs,
        max_new_tokens = args.max_length_cot,
        temperature=args.temperature,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria)

    if stop_on_confidence:
        # The stop token carries the '%' itself, so it must be kept; the default path
        # trims the trailing EOS instead.
        generate_ids = generate_ids[0][prompt_len:]
    else:
        generate_ids = generate_ids[0][prompt_len:-1]
    infer_res = tokenizer.decode(generate_ids)
    return infer_res


def tokenize(prompt, tokenizer, model_name, tokenizer_args=None):
    if 'instruct' in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        model_input = tokenizer.apply_chat_template(messages, return_tensors="pt", **(tokenizer_args or {})).to('cuda')
    else: # non instruct model
        model_input = tokenizer(prompt, return_tensors='pt', **(tokenizer_args or {}))
        if "input_ids" in model_input:
            model_input = model_input["input_ids"].to('cuda')
    return model_input


def generate(model_input, model, model_name, do_sample=False, output_scores=False, temperature=1.0, top_k=50, top_p=1.0,
             max_new_tokens=100, stop_token_id=None, tokenizer=None, output_hidden_states=False, additional_kwargs=None):

    if stop_token_id is not None:
        eos_token_id = stop_token_id
    else:
        eos_token_id = None

    model_output = model.generate(model_input,
                                  max_new_tokens=max_new_tokens, output_hidden_states=output_hidden_states,
                                  output_scores=output_scores,
                                  return_dict_in_generate=True, do_sample=do_sample,
                                  temperature=temperature, top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
                                  **(additional_kwargs or {}))

    return model_output


def generate_model_answer(args, prompt, model, tokenizer, device, do_sample=False, output_scores=False,
                           temperature=1.0,
                           top_p=1.0, max_new_tokens=100, stop_token_id=None, verbose=False):

    model_name = HF_NAMES[args.model_engine]

    model_input = tokenize(prompt, tokenizer, model_name).to(device)

    with torch.no_grad():

        model_output = generate(model_input, model, model_name, do_sample, output_scores, max_new_tokens=max_new_tokens,
                                top_p=top_p, temperature=temperature, stop_token_id=stop_token_id, tokenizer=tokenizer)

    answer = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):])
    output_ids = model_output['sequences'][0][len(model_input[0]):].cpu()
    if output_scores:
        scores = torch.concatenate(model_output['scores']).cpu()  # shape = (new_tokens, len(vocab))
        input_output_ids = model_output['sequences'][0].cpu()
        return answer, input_output_ids, scores, output_ids
    else:
        return answer, output_ids
