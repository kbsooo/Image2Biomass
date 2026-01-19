## Competition Overview

In this competition, your task is to use pasture images to predict five key biomass components critical for grazing and feed management:

- Dry green vegetation (excluding clover)
- Dry dead material
- Dry clover biomass
- Green dry matter (GDM)
- Total dry biomass

Accurately predicting these quantities will help farmers and researchers monitor pasture growth, optimize feed availability, and improve the sustainability of livestock systems.

## Files

**test.csv**

- `sample_id` — Unique identifier for each prediction row (one row per image–target pair).
- `image_path` — Relative path to the image (e.g., `test/ID1001187975.jpg`).
- `target_name` — Name of the biomass component to predict for this row. One of: `Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`, `GDM_g`, `Dry_Total_g`.

The test set contains over 800 images.

**train/**

Directory containing training images (JPEG), referenced by `image_path`.

**test/**

Directory reserved for test images (hidden at scoring time); paths in `test.csv` point here.

**train.csv**

- `sample_id` — Unique identifier for each training sample (image).
- `image_path` — Relative path to the training image (e.g., `images/ID1098771283.jpg`).
- `Sampling_Date` — Date of sample collection.
- `State` — Australian state where sample was collected.
- `Species` — Pasture species present, ordered by biomass (underscore-separated).
- `Pre_GSHH_NDVI` — Normalized Difference Vegetation Index (GreenSeeker) reading.
- `Height_Ave_cm` — Average pasture height measured by falling plate (cm).
- `target_name` — Biomass component name for this row (`Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`, `GDM_g`, or `Dry_Total_g`).
- `target` — Ground-truth biomass value (grams) corresponding to target_name for this image.

**sample_submission.csv**

`sample_id` — Copy from `test.csv`; one row per requested (image, `target_name`) pair.
`target` — Your predicted biomass value (grams) for that `sample_id`.

## What you must predict

For each `sample_id` in `test.csv`, output a single numeric target value in `sample_submission.csv`. Each row corresponds to one (`image_path`, `target_name`) pair; you must provide the predicted biomass (grams) for that component. The actual test images are made available to your notebook at scoring time.

## Citation

Please cite this paper if you are using this dataset for research purposes.

```
@misc{liao2025estimatingpasturebiomasstopview,

      title={Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture},

      author={Qiyu Liao and Dadong Wang and Rebecca Haling and Jiajun Liu and Xun Li and Martyna Plomecka and Andrew Robson and Matthew Pringle and Rhys Pirie and Megan Walker and Joshua Whelan},

      year={2025},

      eprint={2510.22916},

      archivePrefix={arXiv},

      primaryClass={cs.CV},

      url={https://arxiv.org/abs/2510.22916},

}
```

## Evaluation

### Scoring

The model performance is evaluated using a globally weighted coefficient of determination (R²) computed over all (image, target) pairs together.
Each row is weighted according to its target type using the following weights:

- `Dry_Green_g`: 0.1
- `Dry_Dead_g`: 0.1
- `Dry_Clover_g`: 0.1
- `GDM_g`: 0.2
- `Dry_Total_g`: 0.5

This means that instead of calculating R² separately for each target and then averaging, a single weighted R² is computed using all rows combined, with the above per-row weights applied.

### R² Calculation

The weighted coefficient of determination $R_w^2$ is calculated as:

$$R^2_w = 1 - \frac{\sum_j w_j (y_j - \hat{y}_j)^2}{\sum_j w_j (y_j - \bar{y}_w)^2}$$

where $\bar{y}_w = \frac{\sum_j w_j y_j}{\sum_j w_j}$.

\subsection*{Residual Sum of Squares $SS_{\text{res}}$}
Measures the total error of the model's predictions:

$$SS_{\text{res}} = \sum_j w_j (y_j - \hat{y}_j)^2$$

\subsection*{Total Sum of Squares $SS_{\text{tot}}$}
Measures the total weighted variance in the data:

$$SS_{\text{tot}} = \sum_j w_j (y_j - \bar{y}_w)^2$$

\subsection*{Terms}
\begin{itemize}
    \item $y_j$: ground-truth value for data point $j$
    \item $\hat{y}_j$: model prediction for data point $j$
    \item $w_j$: per-row weight based on target type
    \item $\bar{y}_w$: global weighted mean of all ground-truth values
\end{itemize}
