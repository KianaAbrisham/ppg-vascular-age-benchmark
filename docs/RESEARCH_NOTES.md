# Research provenance and implementation notes

This repository consolidates the author's `Wiley-Digital.ipynb` and `Wiley-Radial.ipynb` research notebooks associated with [Neural network models for predicting vascular age from PPG signals: A comparative study](https://doi.org/10.1049/wss2.12103). Original notebook files were preserved separately; their SHA-256 hashes and the refactor's reproduction status are recorded in [`provenance.json`](provenance.json).

## Task and data identity

The related study uses [PWDB](https://zenodo.org/records/3275625), an in-silico dataset of virtual adults. This implementation classifies the six age labels **25, 35, 45, 55, 65, and 75**. These are categorical simulation labels, not direct measurements of continuous vascular or biological age.

The source notebook filenames disagree with the artery files loaded inside them. Set `--site` from a verified data export and its metadata, not the notebook filename. Each run expects one waveform per subject from one artery. Use separate runs for different arteries; repeated records from a subject must not be assigned to independent train/test partitions.

## Changes from the notebooks

- **Shared evaluation:** seeded stratified folds replace independent, unseeded shuffles. All selected models use the same train, validation, and test subject IDs.
- **Explicit preprocessing:** the default spectrogram window is Hann. CNN2D normalization is fitted only on inner training subjects, and VGG16 uses its own input preprocessing convention.
- **Data alignment:** strict subject-ID joins replace positional assumptions and truncation. Labels must belong to the fixed six-class set.
- **Input length:** a configured fixed length replaces dataset-wide implicit padding length. Shorter signals are zero-padded; longer signals are rejected without silent cropping.
- **Execution and reuse:** command-line entry points, saved preprocessing, native Keras checkpoints, and automated checks replace dependence on notebook execution state.

These corrections and modeling choices can change the results. Numerical equivalence with the publication is not claimed.

## Model definitions

All models use a six-way softmax output and sparse categorical cross-entropy. Age values are mapped to class indices 0–5; regression target scaling is not used.

| Model | Implemented architecture | Adam learning rate |
| --- | --- | --- |
| MLP | Dense widths 128, 128, and 32 with tanh activations | 0.001 |
| CNN1D | Two Conv1D/pooling stages with 32 and 16 filters, kernel size 7; dense widths 128, 64, 32, and 16 | 0.001 |
| CNN2D | Two Conv2D/pooling stages with 32 filters and kernel size 7; dense widths 64, 32, and 16 | 0.001 |
| VGG16 | VGG16 with `include_top=False`, flattening, and a new six-class output layer | 0.0001 |

The research defaults are a maximum of 100 epochs, early-stopping patience 10, and batch size 32. Validation loss selects the model state used for test evaluation.

`--weights imagenet` affects only the VGG16 backbone. Its new output layer is randomly initialized, and all VGG16 layers remain trainable. `--weights none` initializes the entire VGG16 model randomly. MLP, CNN1D, and CNN2D always use random initialization. The recorded software checks use random initialization for every model.

## Input representations

MLP and CNN1D receive padded waveforms at their supplied amplitude scale. The two spectrogram-based models begin with the same SciPy signal transform:

| Setting | Default |
| --- | --- |
| Waveform length | 2,000 samples; demo mode uses 512 |
| Sampling rate | 500 Hz; configurable, with no resampling |
| Segment length | 76 samples |
| Window | Hann; Hamming and Tukey are optional |
| Overlap | `nperseg // 8` samples |
| Detrending | Constant |
| Representation | Power spectral density, followed by `10 * log10(max(power, 1e-10))` |

The Tukey option uses alpha 0.25. From that log-power representation:

- **CNN2D:** retain the frequency–time grid with one channel. Normalize using one mean and standard deviation fitted on the inner training spectrograms. There is no square resizing for this model.
- **VGG16:** min–max scale each spectrogram independently, resize it to **32 × 32** with first-order interpolation, and repeat it across three channels. Scale to [0, 255], apply RGB-to-BGR ordering, and subtract channel means `[103.939, 116.779, 123.68]`, matching Keras VGG preprocessing. The 32 × 32 size is used in both demo and research modes.

Padding, per-sample amplitude scaling, window choice, and the small VGG16 input size are experiment assumptions. The models share subject splits but use different representations, learning rates, and potentially initialization sources; comparisons must account for those choices.

## Evaluation and interpretation

Research mode uses five stratified outer folds. Within each outer training partition, a stratified 20% validation split supports early stopping. Accuracy, macro F1, and weighted F1 are compared with a training-only majority-class baseline. Saved prediction tables include the six softmax scores, whose calibration has not been assessed.

The demo's 72 artificial waveforms are independent software fixtures, not PWDB records. One-epoch demo results cannot establish model superiority or reproduce the paper. Fold standard deviations are not confidence intervals, and repeated model selection requires nested cross-validation or a separate final test set.

The original final CSV exports, paper-trained checkpoints, and complete final experiment record were unavailable. Saved values in an old notebook do not by themselves establish that every cell belongs to the publication's final experiment. No published metric is presented as a result of this implementation.

Before reporting research scores, verify subject/site correspondence, waveform sample order, sampling rate, and age labels. Retain input hashes, split IDs, configuration, preprocessing, and checkpoints. Hardware and library versions can affect exact numerical reproducibility. See the [validation record](VALIDATION.md) for completed checks. Performance on simulated age categories does not establish performance on wearable or patient recordings.

## References

- [Related publication](https://doi.org/10.1049/wss2.12103)
- [PWDB dataset](https://zenodo.org/records/3275625)
- [SciPy spectrogram documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html)
- [Keras VGG16 architecture and input requirements](https://keras.io/api/applications/vgg/vgg_models/)
- [Keras VGG preprocessing](https://keras.io/api/applications/vgg/vgg_preprocessing/)
- [Keras serialization and saving](https://keras.io/guides/serialization_and_saving/)
