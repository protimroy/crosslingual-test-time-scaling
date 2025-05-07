cd lm-evaluation-harness/

LANGS=(bn de en es fr ja ru sw te th zh)
OUTPUT_DIR=../outputs
mkdir -p $OUTPUT_DIR

for LANG in "${LANGS[@]}"; do
    ### 27B
    OUTPUT_FP=${OUTPUT_DIR}/gemma3_27B-mgsm_direct_${LANG}_0shot

    # for gemma3b, need to change max_model_len as it is originally set to 100K
    lm_eval --model vllm --model_args pretrained=google/gemma-3-27b-it,tokenizer=google/gemma-3-27b-it,dtype=bfloat16,tensor_parallel_size=4,gpu_memory_utilization=0.7,max_model_len=32768 --tasks mgsm_direct_${LANG} --batch_size auto --output_path ${OUTPUT_FP} --log_samples --gen_kwargs "max_gen_toks=32768,temperature=0" --system_instruction "You are a helpful assistant."

    ### 12B
    OUTPUT_FP=${OUTPUT_DIR}/gemma3_12B-mgsm_direct_${LANG}_0shot

    lm_eval --model vllm --model_args pretrained=google/gemma-3-12b-it,tokenizer=google/gemma-3-12b-it,dtype=bfloat16,tensor_parallel_size=4,gpu_memory_utilization=0.7,max_model_len=32768 --tasks mgsm_direct_${LANG} --batch_size auto --output_path ${OUTPUT_FP} --log_samples --gen_kwargs "max_gen_toks=32768,temperature=0" --system_instruction "You are a helpful assistant."
done