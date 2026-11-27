# Text OHP experiments (2026-11-26)

Output Head Patching (OHP) on small open text models, CPU only, no gradient training, no paid services. Scripts and logs are as run. Read the limits before quoting any number.

## 1. Leads (negative result)

- Data: `shaypal5/leadforge-lead-scoring-v1`, intro config (MIT). **Synthetic**, label `converted_within_90_days`. 1000 train (sampled from 3500), 750 valid, 750 test. Opportunity columns excluded because they leak the label.
- Model: SmolLM2-135M-Instruct. Prompt = lead as text + "Yes or No". Ridge on the final hidden state (576 dims), 2-row Yes/No patch (1,152 parameters). Regularization and alpha picked on valid; test used once.
- Test AUC: base 0.549. OHP with 60 / 100 / 200 labels: 0.550 / 0.549 / 0.550 (10 draws each, range 0.53 to 0.57). OHP with 1000: 0.574. Logistic regression on raw columns: 0.51 (N=100), 0.605 (N=1000). Gradient boosting: 0.55 / 0.56.
- Conclusion: no improvement at small N, and it does not beat a plain tabular model. This dataset has a low ceiling (~0.6).
- Mechanical check (fp32): edited model matches the linear score within 2e-5. SmolLM2 ties lm_head to the input embeddings, so the head was untied first; otherwise the edit would also change inputs.

## 2. SaaS taglines (positive but weak evidence)

- Data: `Hanno-Labs/saas-taglines-rated` (CC-BY-4.0): 158 taglines, quality 1-5 rated **by an LLM judge, not people**. Good = 4-5 (50), bad = 1-2 (64), 3s dropped for the main contrast (n=114).
- Evaluation: 5-fold x 4 repeats cross-validation. Layer and regularization chosen only inside each training fold. No separate held-out test (too small). Std across folds 0.07 to 0.09.
- Good vs bad AUC: SmolLM2-135M base 0.41, OHP final layer 0.80. SmolLM2-360M base 0.50, OHP 0.84, linear probe on a chosen middle layer 0.86 (this is a probe, not OHP).
- Good vs everything else (n=158): 135M OHP 0.68, 360M OHP 0.735. Below 0.8.

## Not measured

Human labels or the owner's own taste; an independent test set; Spanish or other domains; a TF-IDF or other classical baseline; generation (these only classify); models above 360M; confidence intervals beyond fold spread. The tagline labels come from one LLM judge, so the patch partly learns that judge's taste.

## Reproduce

`pip install torch transformers scikit-learn pandas pyarrow`. Leads: `feats.py` then `analyze.py` then `ident.py` (expects data under /tmp/lf). Taglines: download `data/taglines.jsonl` to `t.jsonl`, run `ext.py <model> <tag>` then `ev.py <tag>`. Feature arrays are not committed.
