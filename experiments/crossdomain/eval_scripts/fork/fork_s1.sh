cd lm-evaluation-harness/

THINKING=8000
OUTPUT_FP=../outputs/
mkdir -p $OUTPUT_FP

lm_eval --model vllm --model_args pretrained=simplescaling/s1.1-32B,dtype=bfloat16,tensor_parallel_size=4 --tasks s1_fork --batch_size auto --apply_chat_template --output_path ${OUTPUT_FP} --log_samples --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=${THINKING}"  --predict_only