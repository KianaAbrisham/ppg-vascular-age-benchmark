# Validation record

This record distinguishes completed software checks from full research evaluation. Documentation was updated on **2026-09-27**; the recorded local checks and GitHub Actions run below were completed on **2026-09-24**.

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

Original research CSV exports, paper-trained checkpoints, and a complete final experiment record were unavailable. Full PWDB experiments, ImageNet-initialized VGG16 training, and reproduction of the publication's numerical results have not been verified. GPU, Windows, and macOS execution have not been validated. Softmax probability calibration and performance on patient or wearable recordings have not been evaluated.

Before reporting research results, verify subject identity, artery, waveform format, and age labels, then retain the full experiment's configuration, input hashes, subject splits, preprocessing, checkpoints, and held-out predictions. See the [research notes](RESEARCH_NOTES.md) for changes from the source notebooks.
