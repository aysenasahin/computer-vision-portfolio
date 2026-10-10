# Computer Vision Portfolio

Reproducible computer vision and medical imaging experiments in Python, using one public AVT D1 CT example. This portfolio demonstrates volume inspection, physical image geometry checks, pretrained inference, mask inspection, and transparent reporting of limitations.

The engineering contribution is the experiment code: checking NRRD-to-NIfTI conversion, preserving physical coordinates in overlays, separating inference from mask validation, inspecting Slicer segmentation metadata, and producing descriptive region counts with consistency checks. The segmentation model and dataset annotations are external resources. **No model was trained or fine-tuned in this project.**

## Experiments

| Folder | Implemented work | Observed status |
| --- | --- | --- |
| [01_volume_inspection](experiments/01_volume_inspection/README.md) | Read CT geometry and intensity values; save oriented central slices | Completed for D1 |
| [02_aorta_segmentation](experiments/02_aorta_segmentation/README.md) | Verify NRRD-to-NIfTI conversion, run pretrained TotalSegmentator, inspect the existing mask | Inference and technical checks completed; expert anatomical validation pending |
| [03_reference_inspection](experiments/03_reference_inspection/README.md) | Inspect reference NRRD segments/layers, labels, volumes and physical geometry | Completed for D1; anatomical scope not independently matched |
| [04_mask_comparison](experiments/04_mask_comparison/README.md) | Show common, prediction-only and reference-only regions; save volume counts and slice CSV | Descriptive comparison completed; selected-image user review recorded |

Detailed experiment notes are in Turkish. Raw data, model weights, masks, logs and generated figures are deliberately excluded from this repository. No data-derived images are included in this initial publication because redistribution permission has not been verified.

## Data and attribution

Download data separately from the [Aortic Vessel Tree (AVT) dataset page](https://figshare.com/articles/dataset/Aortic_Vessel_Tree_AVT_CTA_Datasets_and_Segmentations/14806362). See the [dataset publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC8760499/) for context. The local example was supplied as AVT / Dongyang / D1; its exact archive version and applicable reuse/redistribution terms have not been independently established here. A source link is not a claim that the local files' license has been verified. Review the source terms before downloading, using or redistributing data and derived figures.

Place the CT and dataset-provided reference at:

```text
data/raw/avt/Dongyang/D1/D1.nrrd
data/raw/avt/Dongyang/D1/D1.seg.nrrd
```

The `.seg.nrrd` file is a dataset-provided reference, not a model output. The pretrained model is [TotalSegmentator](https://github.com/wasserth/TotalSegmentator); credit its [publication](https://doi.org/10.1148/ryai.230024). Model weights are downloaded by the inference tool and kept outside Git. This work uses SimpleITK, NumPy, Matplotlib and the installed TotalSegmentator/nnU-Net dependencies.

## Setup and execution

Run from the project root in PowerShell. The local environment used Python 3.12.0. Direct dependency pins are in `requirements.txt`; the file is not a complete lock of transitive dependencies. Observed inference versions included TotalSegmentator 2.18.0, PyTorch 2.14.1+cpu and nnU-Net 2.8.1. Environment reproduction on other hardware/platforms has not been validated.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These are setup instructions; creating a new environment is unnecessary when the project environment already exists. The following volume-inspection and conversion/inference commands were used locally:

```powershell
.\.venv\Scripts\python.exe experiments/01_volume_inspection/inspect_volume.py --input data/raw/avt/Dongyang/D1/D1.nrrd --output-dir outputs/01_volume_inspection
.\.venv\Scripts\python.exe experiments/02_aorta_segmentation/run_aorta_segmentation.py --input data/raw/avt/Dongyang/D1/D1.nrrd --output-dir outputs/02_aorta_segmentation/D1 --device auto
```

The second script writes `ct.nii.gz`, checks voxel equality and physical geometry, then attempts inference. The first inference attempt encountered a Windows native-module loading restriction before the model ran. Once conversion is verified, **do not repeat that script simply to resume inference**. The successful normal-PowerShell run used:

```powershell
.\.venv\Scripts\python.exe experiments/02_aorta_segmentation/resume_aorta_inference.py
```

The resume script uses the existing verified NIfTI, checks the inference import chain, runs CPU / total / aorta / fast, and creates a new `D1_resume_<timestamp>_<id>` output folder. It logs commands, timings, exit codes and settings; failure stops the run. It does not change Windows security settings. The original loading failure was not observed in the successful normal-PowerShell execution; its exact context-dependent cause is unresolved.

For downstream inspection, set the run directory to the folder produced by your own successful inference. The example below is the recorded D1 run (data and outputs must exist locally):

```powershell
$runDir = 'outputs/02_aorta_segmentation/D1_resume_20261010_180001_c35ec40e'
.\.venv\Scripts\python.exe experiments/02_aorta_segmentation/check_aorta_mask.py --run-dir $runDir
.\.venv\Scripts\python.exe experiments/03_reference_inspection/inspect_reference.py --ct outputs/02_aorta_segmentation/D1/ct.nii.gz --reference data/raw/avt/Dongyang/D1/D1.seg.nrrd --prediction "$runDir/masks/aorta.nii.gz" --output-dir outputs/03_reference_inspection/D1
.\.venv\Scripts\python.exe experiments/04_mask_comparison/compare_masks.py --ct outputs/02_aorta_segmentation/D1/ct.nii.gz --prediction "$runDir/masks/aorta.nii.gz" --reference data/raw/avt/Dongyang/D1/D1.seg.nrrd --output-dir outputs/04_mask_comparison/D1
```

The experiment notes record the actual calls. Downstream scripts read existing files and do not rerun the model. Geometry checks precede direct array overlays; incompatible grids are reported instead of silently compared. Connected components are counted without deletion or automatic correction.

## Observed D1 results

The CT contained 512 × 666 × 251 voxels with approximately 0.7109375 × 0.7109375 × 2.9999992 mm spacing. Conversion preserved voxel values exactly and physical geometry within stated tolerances.

The CPU/fast inference subprocess completed in 446.221 seconds; TotalSegmentator's internal report recorded 436.13 seconds (different timing scopes). The readable binary uint8 prediction contained 176,582 positive voxels, 267.750585 mL of positive-mask volume, and one face-connected component (6-neighbor connectivity). Its CT geometry differences were zero.

The reference contained one 3D layer and one combined segment, with 189,441 positive voxels. Its geometry matched within tolerance. Branches observed by the user are not separately encoded as segment labels. The reference's Aorta terminology does not establish its exact anatomical coverage.

| Descriptive region | Voxels | Volume on CT grid (mL) |
| --- | ---: | ---: |
| Common to both masks | 153,546 | 232.821189 |
| Prediction only | 23,036 | 34.929395 |
| Reference only | 35,895 | 54.427446 |

Region totals matched the two mask totals, and the 251-slice CSV sums matched the volume counts. A 12-slice axial gallery and coronal/sagittal comparisons were produced locally with physical aspect ratio, anatomical directions and color legends.

In a preliminary review of selected images, the user reported visible common areas in the main aorta, extra reference branches toward the neck and lower extremities, a different lower endpoint, and prediction-only areas near the cardiac starting region and some main-aorta edges. This is a user-reported selected-image observation, not expert anatomical validation or an examination of every slice.

## Limits of interpretation

This is a single-example engineering exercise, not clinical validation or evidence of generalizable model performance. The reference vessel tree and predicted aorta do not have independently matched anatomical scope. Prediction-only and reference-only regions are therefore descriptive differences, not automatically false positives or false negatives. **No Dice, IoU or success percentage was calculated.** Mask volume is not vessel-wall volume or wall thickness.

A comparable evaluation would require documented branch inclusion and endpoint rules, lumen/wall definitions, and, if needed, an independently reviewed CT-based reference with separate aorta/branch labels. The reference was not cropped to the prediction, branches were not pruned, and masks were not corrected.

## Repository boundaries

Only source code, dependency pins and shareable documentation are published. `data/`, `outputs/`, `.venv/`, weights, Slicer scenes, diagnostic exports and generated images remain local and ignored. No private research material, credentials or user-specific absolute filesystem paths belong in the repository. See [public data policy](docs/public_data_policy.md) and [roadmap](ROADMAP.md).
