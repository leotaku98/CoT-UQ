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
