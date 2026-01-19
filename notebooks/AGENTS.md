# notebooks/ - Experiments

## OVERVIEW
Numbered experiments (01-26+). Each is self-contained for Kaggle submission. Pair format: `NN_train.py` + `NN_infer.py`.

## WHERE TO LOOK
| Range | Purpose |
|-------|---------|
| 01-05 | EDA, baseline training |
| 06-10 | Physics constraints, aux tasks, pseudo-labeling |
| 11-14 | Hybrid approach (teacher-student) |
| 15-17 | Hyperparameter tuning (Optuna) |
| 18-20 | Optimized baselines |
| 21-26 | OOF calibration, latest experiments |

## LATEST
- `26_train_oof.py`: OOF collection + Ridge calibration
- `26_infer_calibrated.py`: Inference with calibrators

## CONVENTIONS
- `#%%` cell markers (Jupytext)
- Self-contained: no imports from `src/` (Kaggle kernel restriction)
- `IS_KAGGLE` flag for path switching
- `flush()` after each fold (memory cleanup)

## ANTI-PATTERNS
- NEVER submit train notebooks (inference only)
- NEVER hardcode Kaggle paths without `IS_KAGGLE` check
