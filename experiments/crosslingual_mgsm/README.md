# Experiments for Crosslingual Test-Time Scaling

Folder `artifacts/` contains the output artifacts from our experiments for `baselines` models and `s1` models. For `s1`, having `wait` in the folder name indicates extrapolation budget forcing; otherwise, the outputs are from truncation.

Folder `eval/` contains the eval scripts for prompting baselines and s1 models.

We also provide several `util` scripts for extracting the average number of tokens and the accuracy results from lm-eval-harness.