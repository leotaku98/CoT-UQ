# API Self-Probing Refactor — Design Spec

**Date:** 2026-05-25
**Status:** Approved

## Goal

Replace local Llama model loading with black-box API inference (featherless.ai / OpenAI). Refactor the pipeline to run self-probing UQ with 5 context variants across all models defined in `src/model/llm_eval.yaml`.

## Scope

- Self-probing only (P(true) excluded — requires logprobs, incompatible with black-box API)
- 5 variants: `baseline`, `allkeyword`, `keykeyword`, `allstep`, `keystep`
- Models: all entries in `src/model/llm_eval.yaml` (featherless + openai providers)
- Datasets: unchanged (`gsm8k`, `svamp`, `ASDiv`, `hotpotQA`, `2WikimhQA`)

## Files Changed

| File | Change |
|---|---|
| `src/model/api_client.py` | NEW |
| `inference_refining.py` | Simplify — remove token-prob logic |
| `stepuq.py` | Remove AP strategies + p-true; self-probing runs all 5 variants |
| `run_api_pipeline.sh` | NEW — replaces `run_llama_pipeline.sh` |
| `config.py` | Add `--model_id`, `--provider` args; remove `--model_engine`, `--model_path`, `--uq_engine` |

`utils.py`, `analyze_result.py`, `llama2_predict.py` untouched.

## Component Details

### `src/model/api_client.py`

Single public function:

```python
def chat_complete(prompt: str, model_id: str, provider: str, **gen_kwargs) -> str
```

- Reads `FEATHERLESS_API_KEY` + `FEATHERLESS_BASE_URL` or `OPENAI_API_KEY` + `OPENAI_BASE_URL` from environment (via `python-dotenv`)
- Uses `openai` SDK pointed at the provider's base URL
- `gen_kwargs` passes through `temperature`, `max_tokens`, `top_p`
- Raises on API error after 3 retries with exponential backoff

### `inference_refining.py` (simplified)

Per-question loop:
1. Build CoT prompt via `get_cot_prompt()`
2. Call `chat_complete()` → parse with `parse_response_to_dict()` → retry up to `try_times` if invalid
3. Build keyword-extraction prompt via `get_step_exact_tokens()`
4. Call `chat_complete()` → parse with `step_exacts_2_list()` → retry if invalid
5. Write to `output_v1.json`

**Removed:** all `output_scores=True` generation, `probabilities` dict, `match_final_answer_token_ids`, token alignment loops.

**Output schema** (per line in `output_v1.json`):
```json
{
  "id": "...",
  "question": "...",
  "correct answer": "...",
  "type": "...",           // hotpotQA / 2WikimhQA only
  "llm response": "...",
  "llm answer": "...",
  "step-wise keywords": "...",
  "keyword contribution": {...}
}
```

### `stepuq.py` (self-probing only)

Single function `self_probing_uncertainty()`. Iterates over all 5 variants in one pass over `output_v1.json`. For each record × variant, calls `chat_complete()` and writes to the variant's output file.

Variant → context added to prompt:

| Variant | Function | Prompt addition |
|---|---|---|
| `baseline` | — | none |
| `allkeyword` | `extract_keywords()` | all keywords from all steps |
| `keykeyword` | `extract_keykeywords()` | top keywords by contribution score |
| `allstep` | — | full `llm_response` text |
| `keystep` | `extract_keystep()` | single highest-contribution step |

Output files: `<output_path>/confidences/output_v1_self-probing-<variant>.json`

This naming preserves compatibility with `analyze_result.py`, which reads `output_v1_{args.uq_engine}.json`. Running analysis per variant: `python analyze_result.py --uq_engine self-probing-baseline --dataset hotpotQA --output_path output/Qwen2.5-7B-Instruct/hotpotQA/`

Each output record:
```json
{
  "question": "...",
  "correct answer": "...",
  "llm answer": "...",
  "confidence": 0.82,
  "probing response": "..."
}
```

### `run_api_pipeline.sh`

```bash
DATASET=$1
for each model in src/model/llm_eval.yaml:
    MODEL_NAME = last segment of model id (e.g. Qwen2.5-7B-Instruct)
    OUTPUT_PATH = output/${MODEL_NAME}/${DATASET}/

    python inference_refining.py \
        --model_id <id> --provider <provider> --dataset ${DATASET} \
        --output_path ${OUTPUT_PATH} --try_times 20

    python stepuq.py \
        --model_id <id> --provider <provider> --dataset ${DATASET} \
        --output_path ${OUTPUT_PATH} --try_times 5
done
```

YAML parsing done inline with `python -c "import yaml; ..."` — no extra shell deps.

### `config.py`

Arguments to add:
- `--model_id` (replaces `--model_engine` + `--model_path`)
- `--provider` choices: `featherless`, `openai`

Arguments to remove:
- `--model_engine`, `--model_path`

`--uq_engine` is kept (still used by `analyze_result.py` to locate the confidence file).

Gen kwargs (`temperature`, `max_new_tokens`, `top_p`) remain as CLI args and are passed through to `chat_complete()`.

## Output Structure

```
output/
  Qwen2.5-7B-Instruct/
    hotpotQA/
      output_v1.json
      output_v1_w_labels.json       ← written by analyze_result.py
      error_questions/
        output_v1.json
      confidences/
        output_v1_self-probing-baseline.json
        output_v1_self-probing-allkeyword.json
        output_v1_self-probing-keykeyword.json
        output_v1_self-probing-allstep.json
        output_v1_self-probing-keystep.json
  Meta-Llama-3.1-8B-Instruct/
    hotpotQA/
      ...
```

## Dependencies

Add to `requirement.txt`:
- `openai>=1.0.0`
- `python-dotenv`
- `pyyaml`
