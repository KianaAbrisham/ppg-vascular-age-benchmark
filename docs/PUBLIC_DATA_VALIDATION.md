# Public PWDB evaluation — 2026-09-28

Six full five-fold evaluations were completed using all 4,374 virtual PWDB subjects per site: MLP, CNN1D and CNN2D at the digital and radial arteries. These are results from the current refactored implementation. VGG16 was not included in this public-data run.

| Site | Model | Current accuracy | Fold SD | Macro F1 | Published accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Digital | mlp | 97.23% | 1.11 pp | 0.9722 | 92.7% |
| Digital | cnn1d | 99.18% | 1.34 pp | 0.9917 | 99.4% |
| Digital | cnn2d | 99.59% | 0.19 pp | 0.9959 | 99.6% |
| Radial | mlp | 96.50% | 1.31 pp | 0.9651 | 95.3% |
| Radial | cnn1d | 99.36% | 0.78 pp | 0.9936 | 99.3% |
| Radial | cnn2d | 99.61% | 0.37 pp | 0.9961 | 99.6% |

Accuracy and F1 are arithmetic means over the five held-out folds. SD uses `ddof=1` and is reported in percentage points; it is not a confidence interval. Published values are from the related paper and provide context, not a controlled comparison between identical experiments. CNN2D accuracy at both sites rounds to the paper's reported 99.6%; that agreement does not establish full-paper reproduction.

## Protocol

- Official PWDB v0.2; explicit subject-ID matching; one pulse per subject and site.
- Five shared stratified outer folds; 20% of each outer training partition reserved for stratified validation; seed 42.
- Raw 500 Hz waveforms padded to 487 samples. No cropping or resampling.
- Up to 100 epochs, early-stopping patience 10, batch size 32; best validation weights restored.
- CNN2D uses a Hann log-power spectrogram with `nperseg=76`; normalization is fitted only on the inner training subjects.
- All three evaluated models use random initialization. No hyperparameters were changed in response to test-fold scores.

The models share the same partitions. Each subject appears in exactly one outer test fold in each model/site experiment. Age and subject ID are not waveform features. These labels are simulated age categories, not independently measured vascular age.

## Run it

```bash
python tools/prepare_public_pwdb.py --source data/pwdb-source --output data/pwdb --download
python train.py --signals data/pwdb/deep-learning/digital/signals.csv --targets data/pwdb/deep-learning/targets_age.csv --site digital --models mlp cnn1d cnn2d --length 487 --fs 500 --nperseg 76 --window hann --folds 5 --epochs 100 --patience 10 --batch-size 32 --weights none --seed 42 --output runs/pwdb-digital
python train.py --signals data/pwdb/deep-learning/radial/signals.csv --targets data/pwdb/deep-learning/targets_age.csv --site radial --models mlp cnn1d cnn2d --length 487 --fs 500 --nperseg 76 --window hann --folds 5 --epochs 100 --patience 10 --batch-size 32 --weights none --seed 42 --output runs/pwdb-radial
```

Use a new output directory on reruns. All 30 neural fold checkpoints, histories, preprocessing settings, split IDs and softmax outputs were retained in the validation archive. Large checkpoint files are not committed to Git.

## Evidence and checks

- [Input mapping and checksums](public_data/conversion_manifest.json).
- [Unrounded summary](public_data/summary.csv) and [all fold metrics](public_data/fold_metrics.csv).
- [Held-out predictions, compressed CSV](public_data/held_out_predictions.csv.gz): one row per subject, with shared fold ID and all six model/site predictions. `pandas.read_csv` reads the gzip file directly.
- [Independent checks](public_data/verification.json): no train/validation/test subject overlap; complete out-of-fold coverage; target matching; probability integrity; all metrics recomputed; CNN2D training-only scaling verified; saved inference replayed on 16 held-out subjects from every fold.

Execution used Linux CPU, Python 3.12.14, TensorFlow 2.20.0, Keras 3.11.3, NumPy 2.3.5, Pandas 2.2.3, SciPy 1.17.0 and scikit-learn 1.8.0. Model code was taken from commit `d447edd0cfe2e39335a8e3ff8af05223b84d506f`; this update adds data access and evidence without changing model training code.

## Scope

The original notebooks use different shuffling and fold construction and contain multiple experiment variants. The refactoring also documents preprocessing changes. Consequently these results are a public-data evaluation of the current implementation, not a claim that all original experiments have been recreated. No patient, wearable or external-dataset validation was performed. Repeated model selection requires an independent final test or nested evaluation.

See [source data and attribution](PUBLIC_DATA.md), [implementation notes](RESEARCH_NOTES.md), and the [related publication](https://doi.org/10.1049/wss2.12103).
