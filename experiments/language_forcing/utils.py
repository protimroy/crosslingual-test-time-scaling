import json
import csv
import re
import pathlib
from transformers import AutoTokenizer
from typing import Optional, Tuple
from lingua import LanguageDetectorBuilder

MODEL_NAME_FOR_TOKENIZER = "simplescaling/s1.1-32B"
RESULTS_ROOT = "combined_en"

MODEL_SIZES = ["32B", "14B", "7B"]
MODEL_SUBFOLDERS = [f"s1.1-{sz}" for sz in MODEL_SIZES]
EXPERIMENTS = ["combined_en_only"]
LANGUAGES = ["bn","de","en","es","fr","ja","ru","sw","te","th","zh"]

CSV_FILENAME = "s1_infer_8000.csv"

FINAL_ANSWER_REGEX = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)
BOXED_REGEX = re.compile(r"\\boxed\s*\{\s*([^}]+)\s*\}")

def extract_reasoning_and_answer(full_output: str) -> Tuple[str, str]:
    match = FINAL_ANSWER_REGEX.search(full_output)
    if not match:
        return full_output.strip(), ""
    reasoning = full_output[: match.start()].strip()
    final_answer_str = match.group(1).strip()
    return reasoning, final_answer_str

# def parse_boxed_answer(answer_str: str) -> Optional[float]:
#     match = BOXED_REGEX.search(answer_str)
#     if not match:
#         return None
#     inside_box = match.group(1).strip()
#     try:
#         return float(inside_box)
#     except ValueError:
#         return None

def parse_boxed_answer(text: str) -> str:
    # First try to find boxed answer
    boxed_matches = BOXED_REGEX.findall(text)
    if boxed_matches:
        # Get the last boxed match
        raw_answer = boxed_matches[-1].strip()
        # Remove LaTeX commands and non-numeric parts
        cleaned = re.sub(r'\\[a-zA-Z]+|\{.*?\}', '', raw_answer)  # Remove LaTeX
        cleaned = re.sub(r'[^\d.,\-]', '', cleaned)  # Keep digits, . , and -
        cleaned = cleaned.replace(',', '.')  # Handle commas as decimal points
        # Keep only the first valid float-like number
        match = re.search(r'-?\d+(\.\d+)?', cleaned)
        if match:
            return float(match.group(0))
    return None

def parse_numeric_answer(answer_str: str) -> Optional[float]:
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", answer_str)
    if not nums:
        return None
    return float(nums[0])

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FOR_TOKENIZER)

    detector = LanguageDetectorBuilder.from_all_languages().build()

    all_results = []

    for model_subdir in MODEL_SUBFOLDERS:
        model_path = pathlib.Path(RESULTS_ROOT) / model_subdir
        if not model_path.is_dir():
            print(f"[WARNING] Model folder not found: {model_path}")
            continue

        for experiment in EXPERIMENTS:
            experiment_path = model_path / experiment
            if not experiment_path.is_dir():
                print(f"[WARNING] Experiment folder not found: {experiment_path}")
                continue

            for lang_code in LANGUAGES:
                csv_path = experiment_path / lang_code / CSV_FILENAME
                if not csv_path.is_file():
                    print(f"[WARNING] File not found: {csv_path}")
                    continue

                reasoning_lengths = []
                num_correct = 0
                num_total = 0

                lid_counter_file = {}
                total_lid_samples_file = 0

                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        full_output = row["output"]
                        gold_answer_str = row["gold_answer"]

                        reasoning, final_answer_str = extract_reasoning_and_answer(full_output)

                        tokens = tokenizer(reasoning, add_special_tokens=False)["input_ids"]
                        reasoning_lengths.append(len(tokens))

                        final_answer_float = parse_boxed_answer(final_answer_str)
                        if final_answer_float is None:
                            final_answer_float = parse_numeric_answer(final_answer_str)

                        gold_answer_float = None
                        try:
                            gold_answer_float = float(gold_answer_str)
                        except ValueError:
                            pass

                        if gold_answer_float is not None and final_answer_float is not None:
                            if abs(gold_answer_float - final_answer_float) < 1e-6:
                                num_correct += 1
                        else:
                            if final_answer_str.strip() == gold_answer_str.strip():
                                num_correct += 1

                        num_total += 1

                        cleaned_text = full_output.replace("\n", " ")
                        detected_language = detector.detect_language_of(cleaned_text)
                        if detected_language is None:
                            predicted_lang = "unknown"
                        else:
                            predicted_lang = detected_language.iso_code_639_1.name.lower()

                            if predicted_lang not in LANGUAGES:
                                predicted_lang = "unknown"

                        lid_counter_file[predicted_lang] = lid_counter_file.get(predicted_lang, 0) + 1
                        total_lid_samples_file += 1

                if num_total == 0:
                    print(f"no samples found for {csv_path}. akipping..")
                    continue

                avg_reasoning_length = sum(reasoning_lengths) / len(reasoning_lengths)
                accuracy = num_correct / num_total

                if total_lid_samples_file > 0:
                    lid_distribution = {
                        lang: (count / total_lid_samples_file * 100)
                        for lang, count in lid_counter_file.items()
                    }
                else:
                    lid_distribution = {}

                all_results.append({
                    "model": model_subdir,
                    "experiment": experiment,
                    "language": lang_code,
                    "num_samples": num_total,
                    "avg_reasoning_tokens": avg_reasoning_length,
                    "accuracy": accuracy,
                    "lid_distribution": json.dumps(lid_distribution)
                })

    print("\n===== AGGREGATED SUMMARY =====")
    for r in all_results:
        print(
            f"Model={r['model']} | Exp={r['experiment']} | Lang={r['language']:>2} | "
            f"#Samples={r['num_samples']:>3} | AvgReasoningTokens={r['avg_reasoning_tokens']:.1f} | "
            f"Accuracy={r['accuracy']:.3f} | LID={r['lid_distribution']}"
        )

    summary_csv_path = pathlib.Path(RESULTS_ROOT) / "analysis_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model", "experiment", "language",
            "num_samples", "avg_reasoning_tokens",
            "accuracy", "lid_distribution"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "model": r["model"],
                "experiment": r["experiment"],
                "language": r["language"],
                "num_samples": r["num_samples"],
                "avg_reasoning_tokens": f"{r['avg_reasoning_tokens']:.1f}",
                "accuracy": f"{r['accuracy']:.3f}",
                "lid_distribution": r["lid_distribution"],
            })

    print(f"\n[INFO] Wrote aggregated summary to: {summary_csv_path}")

if __name__ == "__main__":
    main()