# Experiments for Cross-Domain Generalization

We evaluate multilingual cross-domain generalization of s1 models.

### Artifacts
Folder `artifacts/` contains the generated outputs from s1 models on the three benchmarks: [GlobalMMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU), [COPAL-ID](https://arxiv.org/abs/2311.01012), and [FORK](https://aclanthology.org/2023.findings-acl.631/).

### Scripts
Folder `eval_scripts/` contains the eval scripts for each respective benchmarks. 

For instance, for FORK benchmark, run the following scripts for both baseline and s1 models.
```bash
# baseline: Qwen
bash experiments/crossdomain/eval_scripts/fork/fork_qwen.sh

# s1 (think with 8000 tokens)
bash experiments/crossdomain/eval_scripts/fork/fork_s1.sh
```

We also provide `utils` scripts for extracting the answers from models' generations and calculating their accuracies.