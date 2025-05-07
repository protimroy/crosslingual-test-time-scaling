# example script for qwen2.5-7B-Instruct model baseline
cd lm-evaluation-harness/

LANGS=(bn de en es fr ja ru sw te th zh)
OUTPUT_DIR=../outputs

mkdir -p $OUTPUT_DIR
for LANG in "${LANGS[@]}"; do
    ### 0-shot
    OUTPUT_FP=${OUTPUT_DIR}/qwen2.5-7B-mgsm_direct_${LANG}_0shot
    
    lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,tokenizer=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16,tensor_parallel_size=4 --tasks mgsm_direct_${LANG} --batch_size auto --apply_chat_template --output_path ${OUTPUT_FP} --log_samples --gen_kwargs "max_gen_toks=32768,temperature=0"

    ### english cot
    OUTPUT_FP=${OUTPUT_DIR}/qwen2.5-7B-mgsm_en_cot_${LANG}_8shot

    lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,tokenizer=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16,tensor_parallel_size=4 --tasks mgsm_en_cot_${LANG} --num_fewshot 8 --batch_size auto --apply_chat_template --output_path ${OUTPUT_FP} --log_samples --gen_kwargs "max_gen_toks=32768,temperature=0"

    ### native cot
    OUTPUT_FP=${OUTPUT_DIR}/qwen2.5-7B-mgsm_native_cot_${LANG}_8shot

    lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,tokenizer=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16,tensor_parallel_size=4 --tasks mgsm_native_cot_${LANG} --num_fewshot 8 --batch_size auto --apply_chat_template --output_path ${OUTPUT_FP} --log_samples --gen_kwargs "max_gen_toks=32768,temperature=0"
done