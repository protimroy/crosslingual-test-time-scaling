# extract MGSM accuracy scores from lm-eval-harness for a particular model evaluation 
import pathlib
import json

# data_dir = pathlib.Path("crosslingual-test-time-scaling/experiments/crosslingual_mgsm/artifacts/baselines")
# artifact = "qwen2.5-14B-mgsm_direct_*_0shot"
# model_name = "Qwen__Qwen2.5-14B-Instruct"

data_dir = pathlib.Path("crosslingual-test-time-scaling/experiments/crosslingual_mgsm/artifacts/s1")
artifact = "s1.1-32B-wait-mgsm_direct_*_think8000_wait1"
model_name = "simplescaling__s1.1-32B"

results = set()
for result_dir in data_dir.glob(artifact):
    for result_fp in (result_dir / model_name).glob("results_*.json"):
        with open(result_fp) as rf:
            D = json.load(rf)
            for task_name in D["results"]:
                results.add((task_name, D["results"][task_name]["exact_match,flexible-extract"]))
results = list(results)

results.sort(key=lambda e: e[0]) # sort language name alphabetically
assert len(results) == 11, len(results)

print(f"{artifact=}")
results = [round(float(e[1])*100, 2) for e in results]
print("results:", results)
print("avg:", sum(results)/len(results))

