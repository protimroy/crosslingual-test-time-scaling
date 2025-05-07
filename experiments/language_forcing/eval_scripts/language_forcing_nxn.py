import argparse
import pathlib
import csv
import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

LANG_WAIT_TOKENS = {
    "bn": "অপেক্ষা করুন",
    "de": "Warte",
    "en": "Wait",
    "es": "Espera",
    "fr": "Attendez",
    "ja": "待って",
    "ru": "Подождите",
    "sw": "Subiri",
    "te": "వాహించు",
    "th": "รอ",
    "zh": "等待",
}

LANG_PREFIX_TOKENS = {
    "en": "Okay, let me try to figure this out.",
    "bn": "আচ্ছা, সমাধান করার চেষ্টা করি।",
    "de": "Okay, ich versuche, das herauszufinden.",
    "es": "Bien, déjame intentar resolver esto.",
    "fr": "D'accord, laissez-moi essayer de résoudre ça.",
    "ja": "よし、解いてみよう。",
    "ru": "Хорошо, давайте попробуем разобраться.",
    "sw": "Sawa, acha nijaribu kutatua hili.",
    "te": "సరే, దీన్ని పరిష్కరించడానికి ప్రయత్నిస్తాను.",
    "th": "โอเค เดี๋ยวฉันลองแก้ปัญหานี้ดู",
    "zh": "好的，让我来试着解答。",
}

LANG_SYSTEM_PROMPT = {
    "en": "You are a helpful assistant. You must think and answer only in English.",
    "bn": "তুমি একজন বুদ্ধিমান সহকারী, শুধুমাত্র বাংলায় চিন্তা ও উত্তর দাও।",
    "de": "Du bist ein hilfreicher Assistent. Du musst nur auf Deutsch denken und antworten.",
    "es": "Eres un asistente útil. Debes pensar y responder solo en español.",
    "fr": "Tu es un assistant utile. Tu dois réfléchir et répondre uniquement en français.",
    "ja": "あなたは有能なアシスタントです。日本語でのみ考え、回答してください。",
    "ru": "Вы полезный помощник. Вы должны мыслить и отвечать только на русском языке.",
    "sw": "Wewe ni msaidizi mwenye msaada. Unapaswa kufikiria na kujibu tu kwa Kiswahili.",
    "te": "నువ్వు సహాయకరమైన సహాయకుడు. నువ్వు తెలుగులోనే ఆలోచించి సమాధానాలు ఇవ్వాలి.",
    "th": "คุณเป็นผู้ช่วยที่เป็นประโยชน์ คุณต้องคิดและตอบเฉพาะเป็นภาษาไทยเท่านั้น",
    "zh": "你是一个有用的助手，你必须只用中文进行思考和回答。"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="simplescaling/s1.1-32B")
    parser.add_argument("--device-number", type=int, default=2)
    parser.add_argument("--max-tokens-thinking", type=int, default=8000)
    parser.add_argument("--experiment", type=str, default="combined",
                        choices=["wait", "prefix", "combined", "system"])
    parser.add_argument("--num-ignore", type=int, default=1)
    parser.add_argument("--lang-subsets", type=str,
                        default="bn,de,en,es,fr,ja,ru,sw,te,th,zh")
    parser.add_argument("--output-dir", type=str, default="processed/32B_combined")
    parser.add_argument("--dataset-name", type=str, default="juletxara/mgsm")
    parser.add_argument("--num-samples", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()

# CUDA_VISIBLE_DEVICES=2 python language_forcing_nxn.py --model-name simplescaling/s1.1-14B --device-number 1 --lang-subsets sw --output-dir processed/14B_NxN --batch-size 2 --experiment combined
def count_existing_lines(file_path: pathlib.Path) -> int:
    """Return the count of non-empty lines in the file, or 0 if it doesn't exist."""
    if not file_path.is_file():
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def prepare_prompts(dataset, system_text, prefix, start_idx, batch_size):
    """Prepare a batch of prompts."""
    batch_prompts = []
    batch_answers = []
    
    end_idx = min(start_idx + batch_size, len(dataset))
    for i in range(start_idx, end_idx):
        prompt_text = dataset[i]["question"]
        gold_answer = dataset[i]["answer_number"]
        
        system_user_prompt = (
            "<|im_start|>system\n" + system_text + "<|im_end|>\n"
            "<|im_start|>user\n" + prompt_text + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        full_prompt = system_user_prompt + "<|im_start|>think" + prefix
        
        batch_prompts.append(full_prompt)
        batch_answers.append((prompt_text, gold_answer))
        
    return batch_prompts, batch_answers, end_idx


def main():
    args = parse_args()

    # Initialize model with bfloat16 precision
    model = LLM(args.model_name, tensor_parallel_size=args.device_number, dtype="bfloat16", gpu_memory_utilization=0.99, max_model_len=27440)
    tok = AutoTokenizer.from_pretrained(args.model_name)

    lang_subsets = [lang.strip() for lang in args.lang_subsets.split(",")]
    reasoning_langs = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
    reasoning_langs = ["th"]
    for lang_code in lang_subsets:
        for reasoning_lang in reasoning_langs:
            mgsm_dataset = load_dataset(args.dataset_name, lang_code, split="test").select(
                range(args.num_samples)
            )

            if args.experiment == "wait":
                prefix = ""
                wait_token = LANG_WAIT_TOKENS.get(lang_code, "Wait")
                system_text = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            elif args.experiment == "prefix":
                prefix = LANG_PREFIX_TOKENS.get(lang_code, "Okay, let me try to figure this out.")
                wait_token = LANG_WAIT_TOKENS.get(lang_code, "Wait")
                system_text = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            elif args.experiment == "system":
                prefix = ""
                wait_token = LANG_WAIT_TOKENS.get(lang_code, "")
                system_text = LANG_SYSTEM_PROMPT.get(lang_code, "You are a helpful assistant.")
            else:
                prefix = LANG_PREFIX_TOKENS.get(reasoning_lang, "Okay, let me try to figure this out.")
                wait_token = LANG_WAIT_TOKENS.get(reasoning_lang, "Wait")
                system_text = LANG_SYSTEM_PROMPT.get(reasoning_lang, "You are a helpful assistant.")

            print(args.experiment, wait_token, system_text, prefix)
            out_dir = pathlib.Path(args.output_dir) / lang_code
            out_dir.mkdir(parents=True, exist_ok=True)
            result_jsonl_fp = out_dir / f"s1_infer_{args.max_tokens_thinking}_{reasoning_lang}.jsonl"
            result_csv_fp = out_dir / f"s1_infer_{args.max_tokens_thinking}_{reasoning_lang}.csv"

            existing_count = count_existing_lines(result_jsonl_fp)
            if existing_count > 0:
                print(f"[INFO] Found {existing_count} lines for {lang_code}; resuming...")

            dataset_size = len(mgsm_dataset)
            if existing_count == dataset_size:
                continue

            remaining_dataset = mgsm_dataset.select(range(existing_count, dataset_size))

            

            json_mode = "a" if existing_count > 0 else "w"
            csv_mode = "a" if existing_count > 0 else "w"
            wf = open(result_jsonl_fp, json_mode, encoding="utf-8", buffering=1)
            csv_f = open(result_csv_fp, csv_mode, newline="", encoding="utf-8")

            writer = csv.DictWriter(csv_f, fieldnames=["language", "prompt", "output", "gold_answer"])
            if existing_count == 0:
                writer.writeheader()

            stop_token_ids_im_end = tok("<|im_end|>")["input_ids"]
            stop_token_ids_think = tok("<|im_start|><|im_end|>")["input_ids"]

            start_idx = 0
            with tqdm(total=len(remaining_dataset), desc=f"Processing {lang_code} with reasoning lang {reasoning_lang}") as pbar:
                while start_idx < len(remaining_dataset):
                    # Prepare batch of prompts
                    batch_prompts, batch_answers, next_idx = prepare_prompts(
                        remaining_dataset, system_text, prefix, start_idx, args.batch_size
                    )
                    
                    # First thinking phase in batch
                    sampling_params = SamplingParams(
                        max_tokens=args.max_tokens_thinking,
                        min_tokens=0,
                        stop_token_ids=stop_token_ids_think,
                        skip_special_tokens=False,
                        temperature=0.0,
                    )
                    outputs = model.generate(batch_prompts, sampling_params=sampling_params)
                    
                    # Process each output in the batch
                    result_prompts = []
                    remaining_tokens_list = []
                    
                    for output, prompt in zip(outputs, batch_prompts):
                        partial_reasoning = output.outputs[0].text
                        remaining_tokens = args.max_tokens_thinking - len(output.outputs[0].token_ids)
                        result_prompts.append(prompt + partial_reasoning)
                        remaining_tokens_list.append(remaining_tokens)
                    
                    # Handle the wait token steps for each item in batch
                    for _ in range(args.num_ignore):
                        next_batch_prompts = []
                        next_remaining_tokens = []
                        
                        for i, (full_prompt, remaining) in enumerate(zip(result_prompts, remaining_tokens_list)):
                            if remaining <= 0:
                                next_batch_prompts.append(None)
                                next_remaining_tokens.append(0)
                                continue
                            next_batch_prompts.append(full_prompt + wait_token)
                            next_remaining_tokens.append(remaining)
                        
                        # Filter out completed prompts
                        active_indices = [i for i, prompt in enumerate(next_batch_prompts) if prompt is not None]
                        if not active_indices:
                            break
                            
                        active_prompts = [next_batch_prompts[i] for i in active_indices]
                        active_remaining = [next_remaining_tokens[i] for i in active_indices]
                        
                        # Process active prompts
                        batch_sampling_params = [
                            SamplingParams(
                                max_tokens=tokens,
                                min_tokens=1,
                                stop_token_ids=stop_token_ids_think,
                                skip_special_tokens=False,
                                temperature=0.0,
                            )
                            for tokens in active_remaining
                        ]
                        
                        batch_outputs = model.generate(active_prompts, sampling_params=batch_sampling_params)
                        
                        # Update results
                        for batch_idx, model_idx in enumerate(active_indices):
                            output = batch_outputs[batch_idx]
                            used_length = len(output.outputs[0].token_ids)
                            remaining_tokens_list[model_idx] -= used_length
                            result_prompts[model_idx] += wait_token + output.outputs[0].text
                    
                    # Final answer phase in batch
                    final_prompts = [prompt + "Final Answer:" for prompt in result_prompts]
                    sampling_params_final = SamplingParams(
                        max_tokens=27440,
                        min_tokens=0,
                        stop_token_ids=stop_token_ids_im_end,
                        skip_special_tokens=False,
                        temperature=0.0,
                    )
                    final_outputs = model.generate(final_prompts, sampling_params=sampling_params_final)
                    
                    # Write results
                    for i, (final_output, (prompt_text, gold_answer)) in enumerate(zip(final_outputs, batch_answers)):
                        full_output = final_prompts[i] + final_output.outputs[0].text
                        record = {
                            "language": lang_code,
                            "prompt": prompt_text,
                            "output": full_output,
                            "gold_answer": gold_answer,
                        }
                        wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                        writer.writerow(record)
                    
                    # Update progress
                    processed_count = next_idx - start_idx
                    pbar.update(processed_count)
                    start_idx = next_idx

            wf.close()
            csv_f.close()


if __name__ == "__main__":
    main()