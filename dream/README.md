# Dream-v0-Instruct-7B Extension

This folder contains the Dream-v0-Instruct-7B extension for the CDG / DIJA-style injection analysis.

## Current completed pieces

- Dream hidden-state recording completed on 400 examples.
- Groups:
  - A: harmful clean
  - B: harmful injected
  - C: neutral injected
  - D: neutral clean
- Hidden probes show near-perfect injection separability from early denoising fractions.
- Strongest shared BA/CD direction:
  - scope: out_mask
  - frac: 0.05
  - layer: 14
  - cos(BA, CD): 0.980
- Baseline Dream B ASR:
  - 50 / 100
  - ASR = 0.50
- Steering result on 30-case subset:
  - alpha=0.0 ASR = 0.30
  - alpha=2.0 ASR = 0.533
  - alpha=4.0 ASR = 0.433
  - alpha=-0.5 ASR = 0.30
  - alpha=-1.0 ASR = 0.333
  - alpha=-2.0 ASR = 0.433

## Main conclusion

Dream contains a clean shared injection direction, especially at output-mask layer 14, but direct residual steering along this direction did not reduce attack success in the tested 30-case subset.

This supports a detection-correction asymmetry: the injection mechanism is easy to detect internally but difficult to correct by direct steering.

## Files intentionally not committed

Raw outputs, `.pt` tensors, model weights, manifests, and API keys are intentionally excluded.

## P11-style graded judge results

After Fang's note, the saved Dream steering outputs were re-scored using the same graded judge from `p11_ablation.py`.

Metrics:
- Higher `valence_score` means safer.
- Higher `specificity_score` means more actionable / worse.
- Higher `collapse_score` means more degenerate output.

Results on the 30-case B subset:

| alpha | mean valence | mean specificity | mean collapse |
|---:|---:|---:|---:|
| 0.0 | 0.691 | 0.213 | 0.713 |
| 2.0 | 0.646 | 0.247 | 0.660 |
| 4.0 | 0.646 | 0.240 | 0.667 |
| -0.5 | 0.602 | 0.300 | 0.520 |
| -1.0 | 0.546 | 0.297 | 0.570 |
| -2.0 | 0.591 | 0.283 | 0.490 |

Conclusion: using the same p11 graded judge, out_mask L14 steering does not improve Dream safety. Baseline alpha=0.0 has the highest valence score. Steering reduces collapse in some cases, but does not improve valence.
