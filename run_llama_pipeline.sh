export PYTHONPATH=./
MODEL_ENGINE=$1
shift
DATASETS=("$@")

TEMP=1.0
TRY_TIMES=5
TEST_START=0
TEST_END=1000

for DATASET in "${DATASETS[@]}"; do
    OUTPUT_PATH="output/${MODEL_ENGINE}/${DATASET}"
    mkdir -p "${OUTPUT_PATH}" "${OUTPUT_PATH}/confidences" "${OUTPUT_PATH}/error_questions"
    echo ""
    echo "======================================================"
    echo " Model  : ${MODEL_ENGINE}"
    echo " Dataset: ${DATASET}"
    echo " Output : ${OUTPUT_PATH}"
    echo "======================================================"

    CUDA_VISIBLE_DEVICES='0' \
    python inference_refining.py --dataset ${DATASET} --model_engine ${MODEL_ENGINE} \
        --temperature ${TEMP} --try_times ${TRY_TIMES} \
        --test_start ${TEST_START} --test_end ${TEST_END}

    CUDA_VISIBLE_DEVICES='0' \
    python stepuq.py --dataset ${DATASET} --model_engine ${MODEL_ENGINE} \
        --uq_engine self-probing --temperature ${TEMP} --try_times 5
done
