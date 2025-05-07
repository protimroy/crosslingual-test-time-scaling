# Experiments for Language Mixing Analysis

We analyze the dominant language and the language mixing patterns exhibited by s1's thinking.


### Artifacts
Folder `artifacts/` contains the analysis results on the s1's language-mixing patterns. 

### Scripts
Folder `eval_scripts/` contains the eval scripts for (1) analyzing the dominant language and (2) analyzing the language mixing patterns of s1 models. You can tweak the `artifact_names` argument to select a different model outputs for analysis.

```bash
# dominant language analysis
python3 language_mixing/eval_scripts/eval_dominant_language.py

# language mixing analysis
python3 language_mixing/eval_scripts/eval_langmix_patterns.py
```