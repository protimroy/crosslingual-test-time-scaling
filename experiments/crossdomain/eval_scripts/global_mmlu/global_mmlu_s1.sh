cd lm-evaluation-harness/

THINKING=8000
OUTPUT_FP=../outputs/
mkdir -p $OUTPUT_FP

lm_eval --model vllm \
    --model_args pretrained=simplescaling/s1.1-32B,dtype=bfloat16,tensor_parallel_size=4 \
    --tasks global_mmlu_zh,global_mmlu_bn,global_mmlu_de,global_mmlu_fr,global_mmlu_en,global_mmlu_ja,global_mmlu_sw \
    --batch_size auto --apply_chat_template --output_path ${OUTPUT_FP} --log_samples \
    --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=${THINKING}"



