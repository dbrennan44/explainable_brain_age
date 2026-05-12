# macOS Apple Silicon Install Notes

These notes document local installation deviations observed on an Apple Silicon
Mac (`osx-arm64`).

## Observed deviations

- `tensorflow==2.13.1` did not install successfully on this machine.
- `tensorflow==2.13.0` installed and worked.
- `antspyx` was not available from `conda-forge` for `osx-arm64` in this setup,
  so it was installed via `pip` and built from source.

## Install

```bash
conda env create -f environment.yml
conda activate explainable-brain-age
```

The project `environment.yml` installs `antspyx`, `antspynet`, and TensorFlow
through `pip` inside the conda environment.

Do not commit local environments or run outputs. `.conda-env/` and `runs/` are
ignored by `.gitignore`.
