# Crosslingual Reasoning through Test-Time Scaling

<p align="center">
    <a href="https://arxiv.org/abs/xxx.xxxx"><img src="https://img.shields.io/badge/arxiv-xxx.xxxx-b31b1b?logo=arxiv" /></a>
</p>

🔥 TL;DR: We show that scaling up thinking tokens of English-centric reasoning language models such as s1 can improve multilingual math reasoning performance. In addition, we analyze language-mixing patterns, performance of different reasoning languages through language forcing, and multilingual cross-domain generalization.

<p align="center">
  <img src="figures/crosslingual_mgsm.jpg" alt="Crosslingual MGSM performance" width="800"/>
</p>

---
## Quick Start

### Installation
We used the modified [lm_eval_harness](https://github.com/EleutherAI/lm-evaluation-harness) from [s1 repository](https://github.com/simplescaling/s1/). We further modify it for supporting our evaluation setup.
```bash
### installation (python 3.10+)
git clone https://github.com/BatsResearch/crosslingual-s1.git
cd crosslingual-s1
pip install -r requirements.txt
cd lm-evaluation-harness
pip install -e .[math,vllm]
```

### Quick Test on 5 MGSM Samples
Here's a quick eval run on 50 MGSM samples using s1.1-3B models with 2000 maximum thinking tokens using truncation strategy (i.e., CoTs will be cut of if they are longer than 2000 tokens.) using 4GPUs.

See [Codes-and-Artifacts](#codes-and-artifacts) for full evaluation scripts.

```bash
# current dir: root (crosslingual-s1)
cd lm-evaluation-harness/

LANG=zh
MODEL=s1.1-3B
THINKING=2000 # truncation: 2000 max thinking token

OUTPUT_FP=../outputs/${MODEL}-mgsm_direct_${LANG}_${THINKING}
lm_eval --model vllm --model_args pretrained=simplescaling/${MODEL},dtype=bfloat16,tensor_parallel_size=4 --tasks mgsm_direct_${LANG} --batch_size auto --apply_chat_template --output_path ${OUTPUT_FP} --log_samples --gen_kwargs max_gen_toks=32768,max_tokens_thinking=${THINKING} --limit 50
```

## Codes and Artifacts
The `experiments/` folder contains our experiment codes and artifacts of models' generations in our experiments. We structure our repository according to the paper sections.
- [crosslingual_mgsm](https://github.com/BatsResearch/crosslingual-s1/tree/main/experiments/crosslingual_mgsm): Crosslingual test-time scaling experiments (Section 4)
- [language_mixing](https://github.com/BatsResearch/crosslingual-s1/tree/main/experiments/language_mixing): Language-mixing experiments (Section 5)
- [language_forcing](https://github.com/BatsResearch/crosslingual-s1/tree/main/experiments/language_forcing): Language-forcing experiments (Section 6)
- [crossdomain](https://github.com/BatsResearch/crosslingual-s1/tree/main/experiments/crossdomain): Cross-domain experiments (Section 7)

Our folder directory is as such
```bash

```

## Citation
```
add bibtex
```