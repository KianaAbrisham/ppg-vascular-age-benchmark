# Validation record

Updated **2026-09-28**. Six full five-fold public-PWDB evaluations are complete; see the [public-data results](PUBLIC_DATA_VALIDATION.md). The earlier local software checks and GitHub Actions run documented below were completed on **2026-09-24** and are retained as historical evidence.

## Local validation

**Environment:** Linux CPU, Python 3.12.14, TensorFlow CPU 2.20.0, Keras 3.11.3. Package versions and run arguments are recorded in [`smoke_results/run.json`](smoke_results/run.json).

| Check | Recorded result |
| --- | --- |
| Automated test suite | 10 test cases passed |
| Subject-ID matching and invalid-input rejection | Passed |
| Disjoint, complete, repeatable stratified partitions | Passed |
| Training-partition scaling and spectrogram-window regression tests | Passed |
| Native model checkpoint save/reload | Passed for `mlp`, `cnn1d`, `cnn2d`, and `vgg16` |
| Actual training | All four models completed one epoch per fold, with two outer folds |
| Reloaded trained predictions compared with saved held-out predictions | Passed for all four models |
| CSV converter round trip | Preserved IDs, waveforms, and age labels |
| Quickstart notebook | Five code cells executed, with saved outputs and no exceptions |
| Notebook schema and Python syntax | Passed |
| Original research notebook hashes | Unchanged |
| Dependency consistency | `pip check` passed |

Evidence:

- [`test_output.txt`](test_output.txt): actual automated test output. A Keras/NumPy deprecation warning was emitted; the suite finished with `OK`.
- [`integration_checks.json`](integration_checks.json): trained-checkpoint and converter checks.
- [`smoke_results/`](smoke_results/): recorded environment, run configuration, and small-run metrics for each model.
- [`../notebooks/01_quickstart.ipynb`](../notebooks/01_quickstart.ipynb): executed training of all four models, MLP inference, and a confusion matrix.
- [`provenance.json`](provenance.json): source notebook hashes and reproduction status.

All recorded training used **72 artificial software-fixture waveforms**, with 12 examples per class, 512 samples per waveform, one epoch per fold, and random initialization. These checks exercise data flow, training, evaluation, and checkpoint reuse. They do not establish convergence, model rankings, published accuracy, physiological simulation quality, or clinical performance.

Paths inside the recorded configuration identify the original validation runtime. Large model checkpoints are excluded from the repository and can be regenerated with the four-model demo command in the README.

## GitHub Actions

The [Checks run on 2026-09-24](https://github.com/KianaAbrisham/ppg-vascular-age-benchmark/actions/runs/36021382575) completed successfully for commit [`003c6b6`](https://github.com/KianaAbrisham/ppg-vascular-age-benchmark/commit/003c6b6194c294b444225c4d34f7b96dc3e59e9d).

The [workflow](../.github/workflows/checks.yml) installs the pinned dependencies on Ubuntu with Python 3.12, then runs:

```bash
python -m unittest discover -s tests -v
python train.py --demo --output runs/ci-demo
```

The unit test suite checks native checkpoint serialization for all four architectures. The workflow's training command selects the default demo model, **MLP only**. Actual training of all four models was covered by the separate local validation above. This record refers to that specific successful run; the repository badge reports the current workflow status.

## Coverage limits

The original research CSV exports, paper-trained checkpoints, and a complete final experiment record remain unavailable. Fresh public-PWDB five-fold evaluations are complete for MLP, CNN1D and CNN2D at both digital and radial sites. Public-data VGG16 training, ImageNet-initialized training, and full-paper numerical reproduction remain unverified. GPU, Windows, and macOS execution have not been validated. Softmax probability calibration and performance on patient or wearable recordings have not been evaluated.

Before reporting research results, verify subject identity, artery, waveform format, and age labels, then retain the full experiment's configuration, input hashes, subject splits, preprocessing, checkpoints, and held-out predictions. See the [research notes](RESEARCH_NOTES.md) for changes from the source notebooks.

## Public-source conversion verified — 2026-09-28

The official PWDB v0.2 waveform archive, haemodynamic targets and provided fiducials were downloaded and verified against publisher checksums. All 4,374 subject IDs were aligned explicitly, all waveform values passed a CSV round-trip check, and the target units and sampling rate were checked against the source documentation. Four additional tests verify shuffled-ID alignment and rejection of duplicate IDs, missing subjects and corrupt cached downloads. See [public data setup](PUBLIC_DATA.md) and the [conversion manifest](public_data/conversion_manifest.json). These checks validate data preparation; they do not establish numerical reproduction of a paper.

## Full public-data evaluations

Six five-fold model/site evaluations are now complete. See the [2026-09-28 report](PUBLIC_DATA_VALIDATION.md) for scores, subject counts and independent checks. VGG16 public-data training and full-paper reproduction remain unverified.

## Execution provenance

The recorded checks were executed with AI coding assistance in a hosted Linux CPU environment. Absolute `/workspace/scratch/` paths in saved logs and configurations identify that historical runtime. They are preserved as execution evidence; the README commands use paths relative to the repository. This documentation and formatting update does not alter the saved metrics, notebook outputs, or source hashes for those runs.
