# Crosslingual Reasoning through Test-Time Scaling

<p align="center">
    <a href="https://arxiv.org/abs/xxx.xxxx"><img src="https://img.shields.io/badge/arxiv-xxx.xxxx-b31b1b?logo=arxiv" /></a>
</p>

🔥 TL;DR: We show that scaling up thinking tokens of English-centric reasoning language models, such as s1 models, can improve multilingual math reasoning performance. We also analyze the language-mixing patterns, effects of different reasoning languages (controlled by our language forcing strategies), and cross-domain generalization (from STEM to domains such as social sciences and cultural benchmarks).

<p align="center">
  <img src="figures/crosslingual_mgsm.jpg" alt="Crosslingual MGSM performance" width="800"/>
</p>

---
## Getting Started

### Installation
We used the modified [lm_eval_harness](https://github.com/EleutherAI/lm-evaluation-harness) from [s1 repository](https://github.com/simplescaling/s1/). We further modify it for supporting our evaluation setup.
```bash
### installation (python 3.10+)
git lfs # check that you have git lfs installed before cloning the repo

git clone https://github.com/BatsResearch/crosslingual-test-time-scaling.git
cd crosslingual-test-time-scaling
pip install -r requirements.txt
cd lm-evaluation-harness
pip install -e .[math,vllm]
```

### Quick Start
Here's a quick eval run on 50 Chinese MGSM samples using s1.1-3B models with 2000 maximum thinking tokens. This should take less than 10 minutes to complete the command on 4 L40S GPUs.

See [Codes-and-Artifacts](#codes-and-artifacts) for full evaluation scripts.

```bash
cd lm-evaluation-harness/

LANG=zh
MODEL=s1.1-3B
THINKING=2000 # truncation strategy: 2000 max thinking token
NGPUS=4
NSAMPLES=50

OUTPUT_FP=../outputs/${MODEL}-mgsm_direct_${LANG}_${THINKING}
lm_eval --model vllm --model_args pretrained=simplescaling/${MODEL},dtype=bfloat16,tensor_parallel_size=${NGPUS} --tasks mgsm_direct_${LANG} --batch_size auto --apply_chat_template --output_path ${OUTPUT_FP} --log_samples --gen_kwargs max_gen_toks=32768,max_tokens_thinking=${THINKING} --limit ${NSAMPLES}

# |    Tasks     |Version|     Filter      |n-shot|  Metric   |   |Value|   |Stderr|
# |--------------|------:|-----------------|-----:|-----------|---|----:|---|------|
# |mgsm_direct_zh|      2|flexible-extract |     0|exact_match|↑  | 0.78|±  |   N/A|
# |              |       |remove_whitespace|     0|exact_match|↑  | 0.00|±  |   N/A|
#
# the MGSM accuracy is 78.0% for this subset of 50 samples.
```

## Codes and Artifacts
The `experiments/` folder contains our experiment **codes** and **artifacts** of models' generations in our experiments. We structure our repository according to the paper sections.
- [crosslingual_mgsm](https://github.com/BatsResearch/crosslingual-test-time-scaling/tree/main/experiments/crosslingual_mgsm): Crosslingual test-time scaling experiments (Section 4)
- [language_mixing](https://github.com/BatsResearch/crosslingual-test-time-scaling/tree/main/experiments/language_mixing): Language-mixing experiments (Section 5)
- [language_forcing](https://github.com/BatsResearch/crosslingual-test-time-scaling/tree/main/experiments/language_forcing): Language-forcing experiments (Section 6)
- [crossdomain](https://github.com/BatsResearch/crosslingual-test-time-scaling/tree/main/experiments/crossdomain): Cross-domain experiments (Section 7)


## Citation
```
add bibtex
```