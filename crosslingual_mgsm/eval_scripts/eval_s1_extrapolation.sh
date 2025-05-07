# eval on s1 with extrapolation budget forcing strategy using 4 GPUs

cd lm-evaluation-harness/

LANGS=(bn de en es fr ja ru sw te th zh)
OUTPUT_DIR=../outputs
MODEL=s1.1-3B
THINKING=8000
N_WAIT=1 # extrapolation: add wait token once to extend thinking

mkdir -p $OUTPUT_DIR
for LANG in "${LANGS[@]}"; do
    OUTPUT_FP=${OUTPUT_DIR}/${MODEL}-wait-mgsm_direct_${LANG}_think${THINKING}_wait${WAIT}
    lm_eval --model vllm --model_args pretrained=simplescaling/${MODEL},dtype=bfloat16,tensor_parallel_size=4 --tasks mgsm_direct_${LANG} --batch_size auto --apply_chat_template --output_path ${OUTPUT_FP} --log_samples --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=${THINKING},thinking_n_ignore=${N_WAIT},thinking_n_ignore_str=Wait"
done