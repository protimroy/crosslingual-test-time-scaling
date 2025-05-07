cd lm-evaluation-harness/

OUTPUT_FP=../outputs/
mkdir -p $OUTPUT_FP

lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-32B-Instruct,dtype=bfloat16,tensor_parallel_size=4 --tasks s1_fork --batch_size auto --apply_chat_template --output_path ${OUTPUT_FP} --log_samples --gen_kwargs "max_gen_toks=32768,temperature=0"  --predict_only

