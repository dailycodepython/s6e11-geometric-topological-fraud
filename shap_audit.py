


import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def run_shap_feature_audit(
    train_df: pd.DataFrame,
    extra_feature_blocks: dict[str, pd.DataFrame] | None = None,
    target_col: str = "is_fraud",
    baseline_auc: float = 0.94396,
    sample_per_class: int = 1000,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Evaluates marginal feature contributions using LightGBM and SHAP TreeExplainer
    on a balanced validation subset.
    """
    base_cols = [col for col in train_df.columns if col not in ["id", target_col]]
    X_full = train_df[base_cols].copy()
    y_full = train_df[target_col].astype(int)

    # Concatenate orthogonal geometric/topological feature blocks
    if extra_feature_blocks:
        for block_name, df_block in extra_feature_blocks.items():
            X_full = pd.concat(
                [X_full.reset_index(drop=True), df_block.reset_index(drop=True)],
                axis=1,
            )
            print(f"[+] Attached block '{block_name}': +{df_block.shape[1]} features")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    # Encode categorical features to integer codes for SHAP compatibility
    X_tr_shap = X_tr.copy()
    X_val_shap = X_val.copy()
    cat_cols = X_tr_shap.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X_tr_shap[col] = X_tr_shap[col].astype("category").cat.codes
        X_val_shap[col] = X_val_shap[col].astype("category").cat.codes

    clf = lgb.LGBMClassifier(
        random_state=42, n_estimators=200, learning_rate=0.05, n_jobs=-1, verbose=-1
    )
    clf.fit(X_tr_shap, y_tr)

    val_preds = clf.predict_proba(X_val_shap)[:, 1]
    new_auc = roc_auc_score(y_val, val_preds)
    print(f"Baseline ROC AUC: {baseline_auc:.5f}")
    print(f"Extended ROC AUC: {new_auc:.5f} (Delta: {new_auc - baseline_auc:+.5f})")

    # Balanced validation sampling for fast CPU TreeExplainer execution
    explainer = shap.TreeExplainer(clf)
    val_sample_idx = np.concatenate([
        np.flatnonzero(y_val.values == 0)[:sample_per_class],
        np.flatnonzero(y_val.values == 1)[:sample_per_class],
    ])
    X_val_sample = X_val_shap.iloc[val_sample_idx]
    shap_values = explainer.shap_values(X_val_sample)

    # Handle different SHAP output shapes across library versions
    if isinstance(shap_values, list):
        shap_fraud = shap_values[1]
    elif shap_values.ndim == 3:
        shap_fraud = shap_values[:, :, 1]
    else:
        shap_fraud = shap_values

    return shap_fraud, X_val_sample


if __name__ == "__main__":
    # Synthetic dummy data for standalone syntax & pipeline verification
    rng = np.random.default_rng(42)
    dummy_train = pd.DataFrame({
        "id": range(5000),
        "amount": rng.exponential(50, 5000),
        "is_fraud": rng.choice([0, 1], size=5000, p=[0.98, 0.02]),
    })
    dummy_ot = pd.DataFrame({"wasserstein_dist": rng.normal(0, 1, 5000)})

    shap_vals, sample_df = run_shap_feature_audit(
        dummy_train, extra_feature_blocks={"Optimal Transport": dummy_ot}, sample_per_class=50
    )
    shap.summary_plot(shap_vals, sample_df, max_display=10, show=False)
