# Reproduction status

## Recovered as historical evidence

The recorded experiment describes deterministic sampling, a 140/40/60 split, the extraction script, ridge fit, row edit, and actual forward-check outputs. Those records were used to write this new explanatory implementation. They are not bundled here and no internal conversation or execution record is part of this deliverable.

## Still required for a numerical reproduction

1. Authorized dataset download and exact original metadata snapshot/order.
2. Original `splits.json`, or independently regenerate and compare the full list of clip IDs. A seed alone does not establish the same split if the input order changes.
3. Original model revision `9cda5c165b28c68a52e9bb542d2b89f0f6963709`, exact processor configuration, full prompt and package lock.
4. Extraction of four frames at 12.5%, 37.5%, 62.5%, 87.5%, final hidden state at the last token, mean per video. Historical configuration: no image splitting; longest edge 512.
5. Calibration-only ridge (lambda 100), validation selection of alpha 1, base Yes-minus-No score plus h @ w.
6. Clip-level scores with IDs, all baselines and the original patch vector. Verify joins by ID rather than positional order.
7. Real model forward checks and a fresh independent holdout evaluation. The historical two-use test cannot be made untouched retrospectively.

The present package stops at a tested mathematical core. It has no media downloader or live model runner and cannot claim to recreate 91.0%.

## Review before publishing

Check scientific claims and validate complete model integration before claiming numerical reproduction. This repository ships only as a method demonstrator. Do not include dataset media or internal records.
