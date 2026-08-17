# Canonical Raw-to-model v2 baseline report

## Decision and legacy freeze

The legacy CNN, scalers, label encoder, golden fixture, parity test, and feature audit are copied
under `legacy_model/`. They remain reproducible historical assets and are not the formal v2
deployment path.

## Dataset and subject split audit

- Source recordings: 405 (`happy/normal/sad`: 135 each).
- The dataset note maps every five consecutive sample numbers to one participant.
- Anonymous grouping uses `subject_id=(sample_number-1)//5` for the clearly mapped range 1–130.
- The note does not clearly map samples 131–134 and its last line says 135–140. Samples 131–135
  are therefore excluded rather than assigned to a guessed participant.
- Complete, unambiguous 30-second windows: 277 (`happy=93`, `normal=97`, `sad=87`) from
  26 subjects.
- Total exclusions: 128 (shorter than 30 seconds or ambiguous subject mapping). No zero padding
  or time stretching is used.
- Every outer fold and inner validation split uses `StratifiedGroupKFold`.
- Train, validation, and test subject sets are asserted disjoint for every fold.
- Each modality scaler is fitted only on the inner training subset.
- No augmentation, pseudo-label, CVAE, or test-fold feedback is used.

The historical 88.89% result is retained only as a historical experiment because it used the
legacy feature contract and random segment-level splitting.

## Canonical feature contract

All training and deployment code imports `eeg_emotion.features.canonical`.

### filtered `(10,4)`

1. Exactly 30 seconds × 512 Hz = 15360 Raw samples.
2. Constant detrend.
3. 50 Hz notch, Q=30, zero-phase SOS filtering.
4. 0.5–45 Hz fourth-order Butterworth bandpass, zero-phase SOS filtering.
5. Ten exact 1536-sample segments.
6. `[mean, std, max, min]` per segment.

### bandpower `(10,4)`

For each filtered 3-second segment, Welch PSD uses Hann, `nperseg=512`,
`noverlap=256`, `nfft=512`, one-sided density scaling. Features are relative
Theta (4–8), Alpha (8–13), Beta (13–30), and Gamma (30–45 Hz) power divided by
total 0.5–45 Hz power.

Bandpower is not constant in the rebuilt dataset. Per-band mean/std:

| Band | Mean | Std |
|---|---:|---:|
| Theta | 0.1923 | 0.0957 |
| Alpha | 0.1485 | 0.1284 |
| Beta | 0.1882 | 0.1297 |
| Gamma | 0.0741 | 0.0738 |

### ATT/MED `(10,4)`

ATT and MED are linearly interpolated onto the 512 Hz signal timeline, then use the same ten
segments and four statistics. Mean edge-coverage missing ratio is 2.60%, maximum 6.66%,
below the fixed 20% rejection threshold. Because the original update rate is about 1 Hz,
approximately 99.81% of 512 Hz aligned points are interpolated; this is recorded explicitly.

## Automatic parity

`tests/test_canonical_feature_extractor.py` uses an anonymous Raw/ATT/MED fixture and verifies
all four `(10,4)` tensors with `rtol=1e-5, atol=1e-6`. Offline dataset generation and captured
CSV conversion call the same extractor. The test passes.

## No-CVAE four-modality CNN baseline

Evaluation: 5-fold subject-grouped outer testing × seeds 42/43/44, with grouped inner validation
and early stopping. Across all 15 outer folds:

- Accuracy: **0.6295 ± 0.0947**
- Macro-F1: **0.6142 ± 0.0988**

Sample-weighted results for each seed:

| Seed | Accuracy | Macro-F1 |
|---:|---:|---:|
| 42 | 0.6462 | 0.6489 |
| 43 | 0.6029 | 0.6000 |
| 44 | 0.6354 | 0.6378 |

Per-class metrics averaged across the three complete grouped-CV runs:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| happy | 0.6160 | 0.5556 | 0.5817 |
| normal | 0.8011 | 0.6770 | 0.7319 |
| sad / negative-frustrated | 0.5129 | 0.6513 | 0.5732 |

Participant-level accuracy varies substantially: mean 0.6321, standard deviation 0.1993,
range 0.2564–0.9583. This variance must be reported and motivates calibration/domain adaptation.

## Current status

The canonical extractor, rebuilt dataset, grouped baseline, predictions, per-fold models/scalers,
per-class metrics, per-subject metrics, and confusion matrix are complete. This baseline is more
credible than the historical 88.89%, but it is not yet a released real-time model. Next comparisons
should use the identical grouped folds for filtered-only, bandpower-only, ATT+MED, filtered+bandpower,
and all-modalities experiments before considering CVAE.
