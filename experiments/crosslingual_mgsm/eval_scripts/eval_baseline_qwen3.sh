cd lm-evaluation-harness/

LANGS=(bn de en es fr ja ru sw te th zh)
OUTPUT_DIR=../outputs
mkdir -p $OUTPUT_DIR

for LANG in "${LANGS[@]}"; do
    OUTPUT_FP=${OUTPUT_DIR}/qwen3_14B-mgsm_direct_${LANG}_0shot

    lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-14B,tokenizer=Qwen/Qwen3-14B,dtype=bfloat16,tensor_parallel_size=4,gpu_memory_utilization=0.7 --tasks mgsm_direct_${LANG} --batch_size auto --output_path ${OUTPUT_FP} --apply_chat_template --log_samples --gen_kwargs "max_gen_toks=32768,temperature=0"
done