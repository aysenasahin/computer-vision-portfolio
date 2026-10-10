# Portfolio workspace guidance

- This workspace contains reproducible computer-vision and medical-image-processing examples using public data.
- Do not add confidential research details, institutional data, private reports, credentials, or identifying metadata.
- Keep source data under `data/raw/`; it is excluded from version control. The local AVT example is `data/raw/avt/Dongyang/D1/D1.nrrd`; its `.seg.nrrd` companion is a dataset-provided reference, not a model output.
- Keep local experiment outputs under `outputs/`; keep environments and model weights out of version control.
- Run experiments from the project root with `.venv\Scripts\python.exe`. For volume inspection, see `experiments/01_volume_inspection/README.md`.
- Document only commands that were run and results that were actually observed. Do not claim model training, automated segmentation, clinical validation, or evaluation unless performed.
