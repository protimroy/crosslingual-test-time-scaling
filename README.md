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
See [Codes-and-Artifacts](#codes-and-artifacts) for full evaluation scripts.

```bash
# current dir: root (crosslingual-s1)
bash experiments/crosslingual_mgsm/eval_scripts/quick_eval_s1.sh # eval on 5 MGSM examples (zh)
```

## Codes and Artifacts
The `experiments/` folder contains our experiment codes and artifacts of models' generations in our experiments. We structure our repository according to the paper sections.
- `crosslingual_mgsm`: Crosslingual test-time scaling experiments (Section 4)
- `language_mixing`: Language-mixing experiments (Section 5)
- `language_forcing`: Language-forcing experiments (Section 6)
- `crossdomain`: Cross-domain experiments (Section 7)

### Citation
```
add bibtex
```