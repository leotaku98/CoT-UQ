# API Self-Probing Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace local Llama model loading with featherless.ai / OpenAI API calls and refactor the pipeline to run self-probing UQ with 5 context variants across all models in `src/model/llm_eval.yaml`.

**Architecture:** A new `src/model/api_client.py` provides a single `chat_complete()` function used by both `inference_refining.py` (CoT generation + keyword extraction) and `stepuq.py` (self-probing, all 5 variants in one pass). A new `run_api_pipeline.sh` reads `src/model/llm_eval.yaml` and loops over all models.

**Tech Stack:** `openai>=1.0.0` (SDK, works with both featherless and OpenAI endpoints), `python-dotenv`, `pyyaml`, `pytest`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirement.txt` | Modify | Add new deps |
| `src/model/api_client.py` | Create | OpenAI-compatible API wrapper |
| `tests/test_api_client.py` | Create | Unit tests for api_client |
| `config.py` | Modify | Replace model/engine args with model_id/provider/top_p |
| `inference_refining.py` | Rewrite | CoT + keyword extraction via API, no token probs |
| `stepuq.py` | Rewrite | Self-probing, 5 variants, all in one pass |
| `run_api_pipeline.sh` | Create | Loops over llm_eval.yaml models |

---

## Task 1: Update dependencies

**Files:**
- Modify: `requirement.txt`

- [ ] **Step 1: Add new dependencies**

Open `requirement.txt` and append these three lines:

```
openai>=1.0.0
python-dotenv
pyyaml
```

- [ ] **Step 2: Install**

```bash
pip install openai python-dotenv pyyaml
```

Expected: no errors, packages available.

- [ ] **Step 3: Commit**

```bash
git add requirement.txt
git commit -m "Add openai, python-dotenv, pyyaml dependencies"
```

---

## Task 2: Create `src/model/api_client.py`

**Files:**
- Create: `src/model/api_client.py`
- Create: `tests/test_api_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/__init__.py` (empty) and `tests/test_api_client.py`:

```python
import pytest
from unittest.mock import MagicMock, patch


def test_chat_complete_featherless(monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_KEY", "fake-key")
    monkeypatch.setenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Step 1: test\nFinal Answer: 42"

    with patch("src.model.api_client.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        from src.model.api_client import chat_complete
        result = chat_complete(
            "some prompt",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            provider="featherless",
            temperature=1.0,
            max_new_tokens=128,
            top_p=0.9,
        )

    assert result == "Step 1: test\nFinal Answer: 42"
    mock_openai_cls.assert_called_once_with(
        api_key="fake-key", base_url="https://api.featherless.ai/v1"
    )


def test_chat_complete_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Final Answer: Paris"

    with patch("src.model.api_client.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        from src.model.api_client import chat_complete
        result = chat_complete(
            "capital of France?",
            model_id="gpt-4o-mini",
            provider="openai",
        )

    assert result == "Final Answer: Paris"
    mock_openai_cls.assert_called_once_with(
        api_key="sk-fake", base_url="https://api.openai.com/v1"
    )


def test_chat_complete_unknown_provider():
    from src.model.api_client import chat_complete
    with pytest.raises(ValueError, match="Unknown provider"):
        chat_complete("prompt", model_id="model", provider="mystery")


def test_chat_complete_retries_on_error(monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_KEY", "fake-key")
    monkeypatch.setenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"

    call_count = 0

    def flaky_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("temporary error")
        return mock_response

    with patch("src.model.api_client.OpenAI") as mock_openai_cls:
        with patch("src.model.api_client.time.sleep"):
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = flaky_create

            from src.model.api_client import chat_complete
            result = chat_complete("prompt", model_id="m", provider="featherless")

    assert result == "ok"
    assert call_count == 3
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /data/haowhuan/PycharmProjects/CoT-UQ && PYTHONPATH=. pytest tests/test_api_client.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `api_client` does not exist yet.

- [ ] **Step 3: Implement `src/model/api_client.py`**

```python
# -*- coding: utf-8 -*-
"""OpenAI-compatible API client for featherless.ai and OpenAI providers."""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def chat_complete(
    prompt: str,
    model_id: str,
    provider: str,
    temperature: float = 1.0,
    max_new_tokens: int = 256,
    top_p: float = 0.9,
    **_kwargs,
) -> str:
    """Send a single-turn chat prompt and return the response text.

    Args:
        prompt: The user message.
        model_id: HuggingFace-style model identifier.
        provider: 'featherless' or 'openai'.
        temperature: Sampling temperature.
        max_new_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling probability.

    Returns:
        The model's response as a string.

    Raises:
        ValueError: If provider is not recognized.
        Exception: Re-raised after 3 failed retries.
    """
    if provider == "featherless":
        api_key = os.environ["FEATHERLESS_API_KEY"]
        base_url = os.environ["FEATHERLESS_BASE_URL"]
    elif provider == "openai":
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = os.environ["OPENAI_BASE_URL"]
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Expected 'featherless' or 'openai'.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=top_p,
            )
            return response.choices[0].message.content
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /data/haowhuan/PycharmProjects/CoT-UQ && PYTHONPATH=. pytest tests/test_api_client.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/model/api_client.py tests/__init__.py tests/test_api_client.py
git commit -m "Add api_client.py with OpenAI-compatible chat_complete for featherless/openai"
```

---

## Task 3: Update `config.py`

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Rewrite `config.py`**

Replace the entire file content:

```python
# -*- coding: utf-8 -*-
"""Shared argument parser for all pipeline entry points."""

import argparse


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments shared across inference_refining, stepuq, and analyze_result."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="maximum number of new tokens to generate per call"
    )
    parser.add_argument(
        "--try_times", type=int, default=20,
        help="retry limit per question for a valid model response"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--dataset", default="gsm8k",
        choices=["ASDiv", "2WikimhQA", "gsm8k", "hotpotQA", "svamp"]
    )
    parser.add_argument("--datapath", default=None, type=str, help="override default dataset path")
    parser.add_argument(
        "--model_id", default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model identifier passed to the API"
    )
    parser.add_argument(
        "--provider", default="featherless", choices=["featherless", "openai"],
        help="API provider"
    )
    parser.add_argument(
        "--uq_engine", default="self-probing-baseline",
        help="used by analyze_result.py to locate confidences/output_v1_<uq_engine>.json"
    )
    parser.add_argument(
        "--output_path", default="output/",
        help="directory for all output files"
    )
    parser.add_argument("--test_start", default="0", help="start index for dataset slice")
    parser.add_argument("--test_end", default="full", help="end index or 'full'")

    return parser.parse_args()


args = parse_arguments()
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
cd /data/haowhuan/PycharmProjects/CoT-UQ && PYTHONPATH=. python -c "from config import args; print(args)"
```

Expected: prints `Namespace(max_new_tokens=256, try_times=20, ...)` with no errors.

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "Refactor config.py: replace model_engine/model_path with model_id/provider, add top_p"
```

---

## Task 4: Rewrite `inference_refining.py`

**Files:**
- Modify: `inference_refining.py`

- [ ] **Step 1: Replace `inference_refining.py`**

```python
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
```

- [ ] **Step 2: Smoke-test the import**

```bash
cd /data/haowhuan/PycharmProjects/CoT-UQ && PYTHONPATH=. python -c "import inference_refining; print('OK')"
```

Expected: `OK` with no errors.

- [ ] **Step 3: Commit**

```bash
git add inference_refining.py
git commit -m "Rewrite inference_refining.py: API-based inference, remove token-prob logic"
```

---

## Task 5: Rewrite `stepuq.py`

**Files:**
- Modify: `stepuq.py`
- Create: `tests/test_stepuq_prompts.py`

- [ ] **Step 1: Write failing tests for `build_self_probing_prompt`**

Create `tests/test_stepuq_prompts.py`:

```python
"""Tests for the self-probing prompt builder variants."""

SAMPLE_CONTRIBUTION = {
    "Step 1": {"Paris": 9, "capital": 7},
    "Step 2": {"France": 8},
}
SAMPLE_RESPONSE = "Step 1: Paris is the capital.\nStep 2: France is the country."
QUESTION = "What is the capital of France?"
ANSWER = "Paris"


def test_baseline_has_no_extra_context():
    from stepuq import build_self_probing_prompt
    prompt = build_self_probing_prompt(QUESTION, ANSWER, "baseline", SAMPLE_CONTRIBUTION, SAMPLE_RESPONSE)
    assert "Paris is the capital" not in prompt
    assert "Paris" in prompt  # appears only as the answer
    assert "Confidence:" in prompt


def test_allkeyword_contains_all_keywords():
    from stepuq import build_self_probing_prompt
    prompt = build_self_probing_prompt(QUESTION, ANSWER, "allkeyword", SAMPLE_CONTRIBUTION, SAMPLE_RESPONSE)
    assert "Paris" in prompt
    assert "capital" in prompt
    assert "France" in prompt


def test_keykeyword_contains_high_score_keywords():
    from stepuq import build_self_probing_prompt
    prompt = build_self_probing_prompt(QUESTION, ANSWER, "keykeyword", SAMPLE_CONTRIBUTION, SAMPLE_RESPONSE)
    # All three keywords have score >= 4, so all should appear
    assert "Paris" in prompt or "France" in prompt


def test_allstep_contains_full_response():
    from stepuq import build_self_probing_prompt
    prompt = build_self_probing_prompt(QUESTION, ANSWER, "allstep", SAMPLE_CONTRIBUTION, SAMPLE_RESPONSE)
    assert "Paris is the capital" in prompt
    assert "France is the country" in prompt


def test_keystep_contains_one_step_only():
    from stepuq import build_self_probing_prompt
    prompt = build_self_probing_prompt(QUESTION, ANSWER, "keystep", SAMPLE_CONTRIBUTION, SAMPLE_RESPONSE)
    # Both steps have avg contribution 8.0; extract_keystep picks the LAST max → Step 2 wins
    # extract_keystep strips the "Step 2: " prefix, so the step label itself is not in the prompt
    assert "France is the country" in prompt
    assert "Paris is the capital" not in prompt  # Step 1 text not included


def test_unknown_variant_raises():
    from stepuq import build_self_probing_prompt
    import pytest
    with pytest.raises(ValueError, match="Unknown variant"):
        build_self_probing_prompt(QUESTION, ANSWER, "unknown", SAMPLE_CONTRIBUTION, SAMPLE_RESPONSE)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /data/haowhuan/PycharmProjects/CoT-UQ && PYTHONPATH=. pytest tests/test_stepuq_prompts.py -v
```

Expected: `ImportError` — `build_self_probing_prompt` not yet defined.

- [ ] **Step 3: Rewrite `stepuq.py`**

```python
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

    Returns:
        Complete prompt string ready to send to the API.

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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /data/haowhuan/PycharmProjects/CoT-UQ && PYTHONPATH=. pytest tests/test_stepuq_prompts.py -v
```

Expected: 6 tests PASSED.

- [ ] **Step 5: Smoke-test the import**

```bash
cd /data/haowhuan/PycharmProjects/CoT-UQ && PYTHONPATH=. python -c "import stepuq; print('OK')"
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add stepuq.py tests/test_stepuq_prompts.py
git commit -m "Rewrite stepuq.py: self-probing only, 5 variants in one pass via API"
```

---

## Task 6: Create `run_api_pipeline.sh`

**Files:**
- Create: `run_api_pipeline.sh`

- [ ] **Step 1: Create the shell script**

```bash
#!/bin/bash
# Usage: sh run_api_pipeline.sh <dataset>
# Reads src/model/llm_eval.yaml and runs inference + self-probing for all models.

export PYTHONPATH=./

DATASET=$1
TRY_TIMES_INF=20
TRY_TIMES_UQ=5

if [ -z "$DATASET" ]; then
    echo "Usage: sh run_api_pipeline.sh <dataset>"
    echo "Datasets: gsm8k svamp ASDiv hotpotQA 2WikimhQA"
    exit 1
fi

# Parse YAML and emit pipe-delimited lines: model_id|provider|temperature|max_new_tokens|top_p
MODEL_LINES=$(python3 -c "
import yaml
with open('src/model/llm_eval.yaml') as f:
    config = yaml.safe_load(f)
for m in config['models']:
    print(f\"{m['id']}|{m['provider']}|{m.get('temperature', 1.0)}|{m.get('max_new_tokens', 128)}|{m.get('top_p', 0.9)}\")
")

while IFS='|' read -r MODEL_ID PROVIDER TEMPERATURE MAX_NEW_TOKENS TOP_P; do
    MODEL_NAME="${MODEL_ID##*/}"
    OUTPUT_PATH="output/${MODEL_NAME}/${DATASET}"

    echo ""
    echo "======================================================"
    echo " Model   : ${MODEL_ID}"
    echo " Provider: ${PROVIDER}"
    echo " Dataset : ${DATASET}"
    echo " Output  : ${OUTPUT_PATH}"
    echo "======================================================"

    python inference_refining.py \
        --dataset "${DATASET}" \
        --model_id "${MODEL_ID}" \
        --provider "${PROVIDER}" \
        --temperature "${TEMPERATURE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --top_p "${TOP_P}" \
        --output_path "${OUTPUT_PATH}" \
        --try_times "${TRY_TIMES_INF}"

    python stepuq.py \
        --dataset "${DATASET}" \
        --model_id "${MODEL_ID}" \
        --provider "${PROVIDER}" \
        --temperature "${TEMPERATURE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --top_p "${TOP_P}" \
        --output_path "${OUTPUT_PATH}" \
        --try_times "${TRY_TIMES_UQ}"

done <<< "$MODEL_LINES"

echo ""
echo "Pipeline complete. Outputs in output/<model>/${DATASET}/"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x run_api_pipeline.sh
```

- [ ] **Step 3: Verify YAML parsing works**

```bash
cd /data/haowhuan/PycharmProjects/CoT-UQ && python3 -c "
import yaml
with open('src/model/llm_eval.yaml') as f:
    config = yaml.safe_load(f)
for m in config['models']:
    print(f\"{m['id']}|{m['provider']}|{m.get('temperature', 1.0)}|{m.get('max_new_tokens', 128)}|{m.get('top_p', 0.9)}\")
"
```

Expected output (4 lines, one per active model):
```
Qwen/Qwen2.5-7B-Instruct|featherless|1|128|0.9
Qwen/Qwen2.5-3B-Instruct|featherless|1|128|0.9
meta-llama/Meta-Llama-3.1-8B-Instruct|featherless|1|128|0.9
zai-org/GLM-4-9B-0414|featherless|1|128|0.9
```

- [ ] **Step 4: Commit**

```bash
git add run_api_pipeline.sh
git commit -m "Add run_api_pipeline.sh: loops over llm_eval.yaml models for all datasets"
```

---

## Task 7: Run full test suite

- [ ] **Step 1: Run all tests**

```bash
cd /data/haowhuan/PycharmProjects/CoT-UQ && PYTHONPATH=. pytest tests/ -v
```

Expected: all tests PASSED (test_api_client + test_stepuq_prompts).

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "All tests passing: API self-probing refactor complete"
```

---

## Running the pipeline

Launch in tmux since this is a long-running job:

```bash
tmux new-session -d -s cot_uq "sh run_api_pipeline.sh hotpotQA 2>&1 | tee ./tmp/cot_uq_hotpotqa.log"
```

Monitor progress:
```bash
tail -f ./tmp/cot_uq_hotpotqa.log
```

Analyze results per variant after the run:
```bash
python analyze_result.py \
    --uq_engine self-probing-keystep \
    --dataset hotpotQA \
    --output_path output/Qwen2.5-7B-Instruct/hotpotQA/
```
