# second_plan

This folder follows the ChatGPT 5.1 review and adds two validation scripts without touching the main pipeline.

- `run_noise_robust.py`: trains with mixed noise types (white/pink/factory) and evaluates cross-noise/SNR to check domain-shift mitigation.
- `run_mds_metric.py`: fuses the subjective dissimilarity matrix as a kernel term (rather than just appending 2D MDS coords) and tests whether the perceptual prior moves the decision boundary.

Both scripts reuse the existing `src` utilities and write results under `outputs/second_plan/`.

Usage examples (run from repo root):

```bash
python second_plan/run_noise_robust.py
python second_plan/run_mds_metric.py
```

Outputs:
- Per-round metrics CSVs and summaries in `outputs/second_plan/`.
- Console prints for quick checks of best hyperparameters (for the metric-kernel run).
