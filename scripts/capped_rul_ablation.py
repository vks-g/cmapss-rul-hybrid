"""Capped-target check: Random Forest trained on RUL clipped at 125 cycles.

Follows the notebook 04 protocol (same grids, same three engine-grouped folds,
seed 42, best candidate refitted on all training engines). Only the training
target changes: RUL is clipped at ``CAP`` for tuning and fitting. Test engines
are still scored at their last cycle against the true, uncapped RUL in
``RUL_FD001.txt``, so the numbers are comparable with ``fd001_test.json``.

Run from the repository root after ``scripts/download_data.py``:

    python scripts/capped_rul_ablation.py
"""

from pathlib import Path

from sklearn.model_selection import GridSearchCV

from rul.data.labels import add_train_rul
from rul.data.load import load_subset
from rul.data.split import group_kfold_splits
from rul.evaluation.official import predict_test_cycles, save_results, score_last_cycle
from rul.models.experiment import COMPARISON_GRIDS, make_comparison_models

ROOT = Path(__file__).resolve().parents[1]
SUBSET, SEED, N_SPLITS, CAP = "FD001", 42, 3, 125
MODEL = "random_forest"
VARIANTS = ("engineered", "raw")


def main() -> None:
    """Tune, refit and score Random Forest on the capped target for both feature sets."""
    train, test, rul_test = load_subset(SUBSET, data_dir=ROOT / "data/raw/CMAPSSData")
    train = add_train_rul(train, cap=CAP)
    folds = list(group_kfold_splits(train, n_splits=N_SPLITS, seed=SEED))
    X_train = train.drop(columns="rul")

    results, predictions = {}, None
    for variant in VARIANTS:
        pipeline = make_comparison_models(engineered=variant == "engineered", seed=SEED)[MODEL]
        search = GridSearchCV(pipeline, COMPARISON_GRIDS[MODEL], cv=folds,
                              scoring="neg_root_mean_squared_error", n_jobs=-1,
                              error_score="raise").fit(X_train, train["rul"])
        last, scores = score_last_cycle(predict_test_cycles(search, test, rul_test))
        results[variant] = {MODEL: {
            "best_params": search.best_params_,
            "cv_rmse": float(-search.best_score_),
            "test": scores,
        }}
        if predictions is None:
            predictions = last[["unit", "cycle", "rul"]].reset_index(drop=True)
        predictions[f"{variant}_{MODEL}"] = last["prediction"].to_numpy()
        print(f"{variant:>10} Random Forest: test RMSE {scores['rmse']:.2f}, "
              f"MAE {scores['mae']:.2f}, NASA {scores['nasa_score']:,.0f}")

    metrics = {
        "subset": SUBSET,
        "protocol": {
            "target": f"training RUL clipped at {CAP} cycles (ablation); test scored on uncapped RUL",
            "train_engines": int(train["unit"].nunique()),
            "test_engines": int(test["unit"].nunique()),
            "scored_at": f"last recorded cycle of each test engine, against RUL_{SUBSET}.txt",
            "tuning": (f"{N_SPLITS} engine-grouped CV folds over all training engines (seed {SEED}), "
                       "scored on the capped target; best candidate refitted on all training engines"),
            "seed": SEED,
        },
        "models": results,
    }
    paths = save_results(f"fd001_test_capped{CAP}", metrics, predictions)
    print({kind: str(path.relative_to(ROOT)) for kind, path in paths.items()})


if __name__ == "__main__":
    main()
