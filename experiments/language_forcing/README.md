# Experiments for Language-Forcing

We perform Language-Forcing to control s1’s reasoning language.

### Artifacts
Folder `artifacts/` contains the output artifacts from our experiments for `language-forcing` for `s1` models. It contains `translated_wait`, `prefix`, `system`, `combined` and `NxN` (cross-lingual) generations in the folder name corresponding to its experiment name and model size.

### Scripts
Folder `eval_scripts/` contains the eval scripts for prompting s1 models for language-forcing.

```bash
bash language_forcing/eval_scripts/language_forcing_experiment.sh
```

### Misc
We also provide several `util` scripts for extracting the average number of tokens and the accuracy results.