cd lm-evaluation-harness/

LANGS=(bn de en es fr ja ru sw te th zh)
OUTPUT_DIR=../outputs
mkdir -p $OUTPUT_DIR

for LANG in "${LANGS[@]}"; do
    ### r1-32B
    OUTPUT_FP=${OUTPUT_DIR}/r1-32B-mgsm_direct_${LANG}_0shot

    lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,tokenizer=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,dtype=bfloat16,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=32768 --tasks mgsm_direct_${LANG} --batch_size auto --apply_chat_template --output_path ${OUTPUT_FP} --log_samples --gen_kwargs "max_gen_toks=32768,temperature=0"

    ### r1-14B
    OUTPUT_FP=${OUTPUT_DIR}/r1-14B-mgsm_direct_${LANG}_0shot

    lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,tokenizer=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,dtype=bfloat16,tensor_parallel_size=4 --tasks mgsm_direct_${LANG} --batch_size auto --apply_chat_template --output_path ${OUTPUT_FP} --log_samples --gen_kwargs "max_gen_toks=32768,temperature=0"
done