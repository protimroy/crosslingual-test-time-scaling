# get average token lengths of outputs
import pathlib
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

##### Parse command line arguments
parser = argparse.ArgumentParser(description='Calculate average token length for model outputs')
parser.add_argument('--model_name', type=str, default="gemma3_12B", help='Name of the model in data_dir')
parser.add_argument('--tokenizer_name', type=str, default="google/gemma-3-12b-it", help='Name of the tokenizer to use')
parser.add_argument('--output_dir', type=str, default="../data/processed/011-pareto", 
                    help='Directory to save analysis results')
parser.add_argument('--data_dir', type=str, default="crosslingual-s1/crosslingual_mgsm/artifacts/baselines",
                    help='Directory containing model outputs')

args = parser.parse_args()

#### load from arguments
model_name = args.model_name
output_dir = pathlib.Path(args.output_dir)
data_dir = pathlib.Path(args.data_dir)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

final_avg_tokens = list()
wf = open(output_dir / f"{model_name}.tsv", "w+")
for result in tqdm(data_dir.glob(f"{model_name}-mgsm*")):
    len_tokens = total = 0
    for MODEL in result.glob("*"):
        for gen_fp in MODEL.glob("samples*"):
            with open(gen_fp) as rf:
                for i, line in enumerate(rf):
                    if i >= 250: break # duplicated generation for MGSM
                    line = json.loads(line)
                    gen = line["filtered_resps"][0]
                    len_tokens += len(tokenizer(gen)['input_ids'])
                    total += 1

    avg_token_len = len_tokens / total
    final_avg_tokens.append(avg_token_len)

    wf.write(f"{result.name}\t{avg_token_len}\n")
    print(f"{result.name}\t{avg_token_len}")
    
wf.write(f"Final Avg Tokens: {sum(final_avg_tokens)/len(final_avg_tokens):.2f}\n")
print(f"Final Avg Tokens: {sum(final_avg_tokens)/len(final_avg_tokens):.2f}")
