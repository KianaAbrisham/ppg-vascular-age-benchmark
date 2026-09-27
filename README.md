# PPG Vascular Age Benchmark

[![Checks](https://github.com/KianaAbrisham/ppg-vascular-age-benchmark/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/KianaAbrisham/ppg-vascular-age-benchmark/actions/workflows/checks.yml)

Compare four neural network families for classifying **six age categories—25, 35, 45, 55, 65, and 75 years—from photoplethysmography (PPG)**. The benchmark uses shared stratified subject splits, model-specific preprocessing, held-out evaluation, and saved-model inference.

**Author:** [Kiana Pilevar Abrisham](https://github.com/KianaAbrisham)  
**Related publication:** [Neural network models for predicting vascular age from PPG signals: A comparative study](https://doi.org/10.1049/wss2.12103) (2025)

Refactored from the author's research notebooks into a TensorFlow/Keras workflow with explicit data checks and automated tests. All four model paths have completed software checks on artificial data. Full reproduction of the publication's numerical results has not been established. The task uses simulated age-category labels; its outputs are not measurements of continuous biological or vascular age.

## Models

| Option | Input and preprocessing | Architecture |
| --- | --- | --- |
| `mlp` | Padded waveform at its supplied amplitude scale | Dense layers with a six-class output |
| `cnn1d` | Padded waveform with one channel | Conv1D, pooling, and dense layers |
| `cnn2d` | Single-channel log-power spectrogram, normalized using training subjects only | Conv2D, pooling, and dense layers |
| `vgg16` | Spectrogram scaled per sample, resized to 32 × 32, and transformed for VGG16 | VGG16 convolutional backbone and a new six-class output |

Every model returns six softmax scores. VGG16 can use ImageNet initialization; all other models train from random initialization. The VGG16 input is **32 × 32 in both demo and research modes**. See the [research notes](docs/RESEARCH_NOTES.md) for exact architectures and preprocessing choices.

## Quick start

Use **Python 3.12** in this repository's folder. Create an isolated environment:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\activate` in Windows Command Prompt. Linux CPU is the validated environment.

Install dependencies, run the tests, and exercise all four model paths:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --demo --models mlp cnn1d cnn2d vgg16 --output runs/demo
```

The demo generates **72 artificial waveforms**, with 12 examples per class and 512 samples per waveform. Each selected model trains for **one epoch in each of two stratified outer folds**, using random initialization. No pretrained weights are downloaded. These fixtures are independent of PWDB, and their scores cannot establish research performance or rank the models meaningfully. Use a new output folder for each training run.

For a shorter workflow check, select only `--models mlp`. Omitting `--models` in demo mode also runs only the MLP.

Reload the first MLP checkpoint and write predictions:

```bash
python predict.py --model-dir runs/demo/mlp/fold_1 --signals runs/demo/demo_input/signals.csv --output runs/demo/new_predictions.csv
```

The CSV contains `subject_id`, `predicted_age_category`, six `p_age_*` columns, and `model_status`. The score columns are model softmax outputs; probability calibration has not been evaluated. This command demonstrates checkpoint reuse on demo inputs. It restores one saved fold model, without fitting an all-data model or combining folds.

The [executed quickstart notebook](notebooks/01_quickstart.ipynb) includes training all four models, an inference example, and a confusion matrix. The [validation record](docs/VALIDATION.md) describes completed checks and GitHub Actions coverage.

## Use research data

The related study uses [PWDB](https://zenodo.org/records/3275625), an in-silico dataset of virtual adults. Original research CSV exports and paper-trained checkpoints are not included. `prepare_data.py` converts existing wide CSV exports; it does not download or process the raw PWDB release automatically.

| File | Required columns | Meaning |
| --- | --- | --- |
| `signals.csv` | `subject_id`, `s0000`, `s0001`, … | One waveform per subject from one artery; sample columns in time order |
| `targets.csv` | `subject_id`, `age` | Exactly one of `25`, `35`, `45`, `55`, `65`, `75` |

The loader matches targets by subject ID. IDs must be unique, and both files must contain exactly the same subjects. It rejects unsupported age labels, empty or constant waveforms, internal missing samples, infinities, and metadata mixed into waveform columns. Shorter waveforms, including trailing empty sample cells, are zero-padded; longer waveforms are rejected without silent cropping.

Convert existing wide exports using the actual source column names:

```bash
python prepare_data.py --signals original_signals.csv --targets original_targets.csv --signal-id-column subject_id --target-id-column subject_id --target-column age --task classification --output data/converted
```

Quote column names containing spaces. Use `--drop-signal-columns` to remove exported indexes or metadata explicitly. If both files lack IDs, use `--assume-row-aligned` only after independently verifying their subject order.

Run the four-model comparison with ImageNet initialization for VGG16:

```bash
python train.py --signals data/converted/signals.csv --targets data/converted/targets.csv --site digital --models mlp cnn1d cnn2d vgg16 --weights imagenet --output runs/digital-01
```

`--weights imagenet` affects only VGG16 and may download its backbone weights. Its new classification layer is randomly initialized, and the entire model is trained. Use `--weights none` for random initialization of all four models. The recorded local and CI checks use `none`; ImageNet-initialized training has not been validated in those checks.

Set `--site` from verified dataset identity: `digital`, `radial`, or `brachial`. The original notebook filenames are not reliable artery labels; see the [provenance notes](docs/RESEARCH_NOTES.md). Use separate runs for different arteries, and do not treat multiple records from one subject as independent subjects.

Research defaults are `--length 2000`, `--fs 500`, `--nperseg 76`, `--window hann`, five folds, a maximum of 100 epochs, early-stopping patience 10, batch size 32, and seed 42. Choose length and sampling rate from the acquisition/export specification; the pipeline does not resample signals. Run `python train.py --help` for all options.

## Evaluation and saved outputs

All models share the same five stratified outer folds. Within each outer training partition, 20% of subjects are reserved for stratified validation. The CNN2D normalization statistics are fitted only on the inner training subjects; other input transforms are per sample or fixed. Age labels are encoded as six class indices, with no regression target scaling.

Early stopping monitors validation loss, and test predictions use the best validation weights. Evaluation reports **accuracy, macro F1, and weighted F1**, alongside a baseline that always predicts the most frequent training class.

Each run saves:

- Configuration, dependency versions, input hashes, and subject IDs for every split.
- Per-fold preprocessing, `.keras` checkpoints, loss histories, metrics, and baseline results.
- Held-out predicted classes and softmax scores, confusion matrices, per-fold metrics, and summaries across folds.

Latency reports the median and 95th percentile of 20 synchronous, batch-one forward passes after three warmups; input preprocessing is excluded. Checkpoint size measures the complete Keras archive, including training state. These measurements depend on the hardware and software environment.

Fold standard deviations describe variation across folds; they are not confidence intervals. Repeated model or hyperparameter selection requires nested cross-validation or a separate final test set. Comparisons reflect each model's representation, initialization, and training settings as well as its architecture.

## Repository guide

| Location | Contents |
| --- | --- |
| [`ppg/`](ppg/) | Data checks, preprocessing, model definitions, training, and inference |
| [`train.py`](train.py), [`predict.py`](predict.py), [`prepare_data.py`](prepare_data.py) | Command-line entry points |
| [`tests/`](tests/) | Input, split, preprocessing, and checkpoint regression tests |
| [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) | Executed artificial-data walkthrough |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | Completed checks, evidence, and coverage limits |
| [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) | Source provenance and implementation choices |
| [`.github/workflows/checks.yml`](.github/workflows/checks.yml) | CPU tests and MLP training demo |

Citation metadata are available in [`CITATION.cff`](CITATION.cff). Performance on simulated age categories alone does not establish performance on patient or wearable recordings.
