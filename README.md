# PPG Vascular Age Model Comparison

Compare four neural network families on six age categories, with identical stratified splits and measured prediction latency.

**Author:** [Kiana Pilevar Abrisham](https://github.com/KianaAbrisham)  
**Related paper:** [Neural network models for predicting vascular age from PPG signals: A comparative study](https://doi.org/10.1049/wss2.12103) (2025)

Refactored from the author's research notebooks. Includes training, saved-model inference,
subject-ID validation, explicit preprocessing, reproducible splits, and automated tests.
Full reproduction of the paper's numerical results has **not** been established.

## Quick start

Use Python **3.12** in this project's folder, with a separate environment for each project.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` in Windows Command Prompt or
`source .venv/bin/activate` on Linux/macOS. Then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --demo --output runs/demo
python predict.py --model-dir runs/demo/mlp/fold_1 --signals runs/demo/demo_input/signals.csv --output runs/demo/new_predictions.csv
```

Demo mode generates artificial waveforms, uses one epoch, and disables pretrained-weight downloads.
It overrides length to 512 and Keras fold count to two. Its scores are **software checks**, not
research or clinical results. Use a new output folder for each run; existing results are protected.
The [quickstart notebook](notebooks/01_quickstart.ipynb) provides the same workflow with recorded outputs.
See [validation](docs/VALIDATION.md) for the checks actually completed.

## Research data

The related study uses [PWDB](https://zenodo.org/records/3275625), an in-silico dataset of virtual adults.
The original local CSV exports are not included. This package does not automatically download or
convert the source dataset. The artificial demo is independent of PWDB.

| CSV | Required columns | Meaning |
| --- | --- | --- |
| signals.csv | subject_id, s0000, s0001, ... | One subject and artery per row; waveform samples only |
| targets.csv | subject_id, age | One of 25,35,45,55,65,75 |

Target rows are joined by ID. ID sets must agree exactly; duplicated subjects, constant signals,
internal missing values, and nonnumeric metadata are rejected. Trailing empty waveform cells are
zero-padded. No waveform is silently cropped. `--length` defaults to 2000; choose a sufficient fixed
length from the acquisition/export specification. `--fs` defaults to 500 Hz; the code does not resample.

Convert existing wide exports using exact original column names:

```bash
python prepare_data.py --signals original_signals.csv --targets original_targets.csv --signal-id-column subject_id --target-id-column subject_id --target-column age --task classification --output data/converted
```

Quote headers containing spaces. Use `--drop-signal-columns` to remove an exported index or metadata
explicitly. If neither CSV has an ID, `--assume-row-aligned` is available only after you independently
verify identical subject ordering. Do not use it when correspondence is unknown.

```bash
python train.py --signals data/converted/signals.csv --targets data/converted/targets.csv --site digital --models mlp cnn1d cnn2d vgg16 --output runs/digital-01
```

Set `--site` from verified dataset identity; filenames alone are insufficient. Use separate runs for
different arteries. Do not pool multiple artery records from the same subject into independent-row
splits. `--weights imagenet` initializes VGG16/ResNet18 from downloaded pretrained weights; use
`--weights none` for random initialization. See `python train.py --help` for settings.

## Evaluation and saved artifacts

Five outer folds, each with an inner 20% validation split. Preprocessing statistics are fitted on inner training subjects only.
Early stopping uses validation loss, and test predictions use the best validation weights.
Models share the same stratified train/validation/test partitions.

Runs save configuration, dependency versions, input hashes, subject splits, history, checkpoints,
preprocessing, held-out predictions, metrics, and plots. A training-only baseline is included.
Regression reports MAE, RMSE, R² and MAPE; classification reports accuracy, macro F1 and weighted F1.
Latency uses three warmups and twenty synchronous batch-one forward passes, excluding preprocessing.
Checkpoint size is serialized file size; Keras archives include training state and are not deployment-only weight size.

`predict.py` restores one fold model and its preprocessing; it does not refit an all-data model or
ensemble folds. Fold standard deviations are not confidence intervals. Repeated model/hyperparameter
selection requires a further independent test set or nested cross-validation.

## Structure and provenance

- `ppg/`: data, preprocessing, model architecture, framework engine, training, inference.
- `train.py`, `predict.py`, `prepare_data.py`: complete entry points.
- `tests/`: alignment, split, window-setting and model-checkpoint regression tests.
- `notebooks/01_quickstart.ipynb`: artificial-data walkthrough.
- `docs/`: source hashes, implementation changes and validation record.
- `.github/workflows/checks.yml`: CPU test and demo workflow.

See [research notes](docs/RESEARCH_NOTES.md) for changes from the notebooks and limitations.
Paper citation metadata are provided in `CITATION.cff`.
