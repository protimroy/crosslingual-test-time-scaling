# eval script for Qwen3 models since lm_eval_harness fails to extract answers from QWen3's output
import json
import pathlib
import re

BOXED_PATTERN = re.compile(r'\\boxed\{([^}]*)\}')
NUMERIC_PATTERN = re.compile(r'(-?\d+(\.\d+)?)')

def extract_last_boxed_answer(text: str) -> str:
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    numeric_str = re.sub(r'[^0-9.-]', '', matches[-1].strip())
    return numeric_str

def parse_numeric(answer_str: str):
    if not answer_str:
        return None

    match = NUMERIC_PATTERN.search(answer_str)
    # Remove any non-numeric characters except for decimal point and negative sign
    if not match:
        return None

    numeric_str = re.sub(r'[^0-9.-]', '', match.group(1))
    try:
        return float(numeric_str)
    except ValueError:
        return None


data_dir = pathlib.Path("crosslingual-s1/crosslingual_mgsm/artifacts/baselines")
MODEL_NAME = "qwen3_14B-mgsm_direct_*_0shot"

lang2acc = {}
for folder in data_dir.glob(MODEL_NAME):
    for model_fp in folder.glob("*"):
        # extract language
        lang = folder.name.split("_")[-2]
        total = count_correct = 0
        for samples_fp in model_fp.glob("samples*"):
            with open(samples_fp) as rf:
                for i, line in enumerate(rf):
                    if i >= 250:
                        break
                    line = json.loads(line)
                    output_text = line['filtered_resps'][0].split("</think>")[-1]
                    output_answer_text = output_text.split("\n\n")[-1].strip()
                    final_answer_str = extract_last_boxed_answer(output_answer_text) or parse_numeric(output_answer_text)
                    if not final_answer_str:
                        # extract from the entire chain-of-thought
                        final_answer_str = extract_last_boxed_answer(output_text) or parse_numeric(output_text)
                    
                    if not final_answer_str:
                        count_correct += 0
                    else:
                        count_correct += abs(float(final_answer_str) - float(line['target'])) < 1e-9
                    total += 1
            acc = round(count_correct / total * 100, 2)
            lang2acc[lang] = acc
            break

lang_accs = [(lang, acc) for lang, acc in lang2acc.items()]
lang_accs.sort(key=lambda x:x[0])
print([acc for lang, acc in lang_accs])
