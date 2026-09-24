# Research provenance

Source notebooks: Wiley-Digital.ipynb, Wiley-Radial.ipynb. Their SHA-256 hashes are in `provenance.json`.
Original notebook bytes were preserved separately; the package is a new implementation copy.

## Implementation changes

Consolidates the two Wiley notebooks. Their filenames disagree with the loaded artery files; the verified data export must determine --site. The window is explicitly Hann, following the paper setting; old calls omitted it. CNN spectrogram normalization is fitted only on training subjects; VGG16 uses ImageNet-compatible input preprocessing. Shared seeded stratified folds replace independent unseeded shuffles.

Across the project: strict subject-ID joins replace positional assumptions/truncation; fixed input
length replaces dataset-wide implicit padding length; regression target scaling occurs inside the
training partition; repeated notebook state is replaced by complete entry points and saved preprocessing.
The corrections and modeling changes can change the reported scores.

Spectrograms use 500 Hz by default, segment length 76, overlap `nperseg // 8`, constant detrending,
PSD output and a 1e-10 power floor before 10·log10. Tukey, when selected explicitly, uses alpha .25.
Zero padding and square image resizing are explicit modeling assumptions.

## Results and limitations

The source studies use 4,374 simulated cardiovascular profiles. This implementation predicts one of six age categories, not continuous biological age.
Performance on simulated profiles does not establish performance on wearable or patient recordings.
The original final CSV exports, paper checkpoints and complete final experiment record were not supplied.
Saved values in the old notebooks cannot certify that every cell is a publication-final experiment.

No published metric is presented as a result of this refactor. Before reporting corrected research
scores, verify subject/site correspondence, sampling rate, target units, and waveform columns; run the
full experiment and retain its configuration, input hashes and split records. Hardware/library versions
can affect exact numerical reproducibility. CPU is the validated target; other devices require validation.

## References

- Paper: https://doi.org/10.1049/wss2.12103
- Dataset: https://zenodo.org/records/3275625
- Signal processing: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
- Keras serialization: https://keras.io/guides/serialization_and_saving/
- VGG preprocessing: https://keras.io/api/applications/vgg/
- ResNet weights: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
