# WSP Output Head Lab

This is a reconstructed method demonstrator, not a recovered release of the original experiment.

A supervised ridge readout merged into the Yes/No output rows of SmolVLM2-256M-Video-Instruct. No gradient descent is used. This is still supervised learning, not learning without labels.

## What is ready

- `wsp.py`: fit ridge on calibration features, apply the two-row edit, score pairs and check the linear identity.
- `tests/test_wsp.py`: synthetic mathematical checks. These do not load SmolVLM2 or reproduce the reported benchmark.
- `results/historical_summary.json`: rounded values reported in the original experiment, clearly separated from new runs.
- `docs/reproduction.md`: required inputs, original method and missing artifacts.

## Quick check

Requires Python 3.10+ and NumPy. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

NumPy >=1.24,<3 is a supported range for these reconstructed checks, not a recovered lockfile from the original run.

## Mechanism

For hidden vector h and original score z = logit(Yes) - logit(No), fit w with ridge regression on calibration features and standardized human scores. Edit the output rows:

```
Yes += alpha * w / 2
No  -= alpha * w / 2
z_patched = z + alpha * h @ w
```

The historical run used 576 features, changing 1,152 parameter entries. Regularization was 100 and alpha was 1, selected using validation. When applying this to a real model, verify token IDs, output-head shape, weight tying and dtype first. This demonstrator does not perform that model integration.

## Historical results, not reproduced here

On 60 KoNViD-1k test clips, 601 pairs with a human-score gap of at least 1.0: base 20.1%, luminance 75.2%, patched 91.0%. Two test evaluations occurred during development. The second design followed inspection of the first test result, so a fresh holdout is needed for cleaner confirmation. Pairs share clips and are not 601 independent observations.

The original forward checks reported maximum error below 0.000006 between the edited model and the external linear score. The synthetic check in this package is not that original check.

## Limits

Four independently processed frames were averaged per clip. No temporal modeling, audio, pacing or complete editing-quality evaluation was established by this task. Transfer to other datasets, prompts and personal editing preferences is unverified. No novelty claim is made.

The original split file, feature arrays, patch vector, exact environment lock and complete machine-readable results have not been recovered into this package. No benchmark reproduction command is claimed to work yet.

## Data and model

Obtain data under the source terms. This package includes no KoNViD videos, frames, MOS tables, client data, credentials or model weights.

- Model: https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct
- Dataset and citation: https://database.mmsp-kn.de/konvid-1k-database.html
- Related project: https://github.com/FranprzDev/reelsmith

The upstream model and dataset retain their own terms. Code in this repository is licensed under MIT. See LICENSE.
