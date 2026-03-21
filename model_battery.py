"""
Model battery for BSI prediction on base_HH_with_risk_factors.
Runs LR, MLP, and XGB with multiple feature combinations and writes test AUC summary.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from sklearn.inspection import permutation_importance
import os
import sys
import joblib

from models import run_lr, run_mlp, run_xgb_fixed, run_xgb, RANDOM_STATE

# Data path
DATA_DIR = 'data/'
DATA_PATH = os.path.join(DATA_DIR, 'base_HH.csv')
TEST_SIZE = 0.2
OUTCOME_COL = 'Positive'

# Columns to drop from features (ids and outcome)
DROP_COLS = ['PatientDurableKey', 'EncounterKey', 'Positive', 'OrderedDateKey', 'OrderedTimeOfDay', 'FalsePositive']


def get_feature_columns(df):
    """Define column groups from dataframe. Only include columns that exist."""
    all_cols = set(df.columns)
    notes_columns = [c for c in df.columns if c.startswith('cn_')]
    risk_columns = [c for c in df.columns if c.startswith('risk_')]
    icd_columns = [c for c in df.columns if c.startswith('icd_')]
    comp_columns = [c for c in df.columns if c.startswith('cmp_')]
    med_columns = [c for c in df.columns if c.startswith('rx_')]
    vitals_names = {'Temp', 'SpO2', 'Pulse', 'Resp', 'HighFever', 'LowFever', 'HighPulse', 'LowOxygen'}
    vitals_columns = [c for c in df.columns if c in vitals_names]
    return {
        'notes_columns': notes_columns,
        'risk_columns': risk_columns,
        'icd_columns': icd_columns,
        'comp_columns': comp_columns,
        'med_columns': med_columns,
        'vitals_columns': vitals_columns,
    }


def get_feature_combinations(col_groups):
    """Return list of (list of column names, feature_type label)."""
    r = col_groups['risk_columns']
    icd = col_groups['icd_columns']
    vit = col_groups['vitals_columns']
    comp = col_groups['comp_columns']
    med = col_groups['med_columns']
    notes = col_groups['notes_columns']
    combos = [
        (r, 'risk'),
        (vit + r, 'vitals_risk'),
        (vit + comp + r, 'vitals_comp_risk'),
        (vit + comp + med + r, 'vitals_comp_med_risk'),
        (vit + comp + icd + r, 'vitals_comp_icd_risk'),
        (vit + comp + med + icd + r, 'vitals_comp_med_icd_risk'),
        (notes, 'notes'),
        (vit + r, 'notes_vitals'),
        (notes + vit + comp + med + icd, 'notes_vitals_comp_med_icd'),
        (notes + vit + comp + med + icd + r, 'notes_vitals_comp_med_icd_risk'),
    ]
    return combos


def _compute_feature_importance(model, model_name, X_te, y_te, feature_names):
    """Return DataFrame with columns: feature, importance (higher = more important)."""
    # XGBoost: use gain from booster; keys can be "f0","f1" or feature names when trained on DataFrame
    if model_name == "xgb":
        try:
            score = model.get_booster().get_score(importance_type="gain")
            if not score:
                # fallback to weight-based importance from sklearn wrapper (always in feature order)
                imp = np.asarray(model.feature_importances_, dtype=float)
            else:
                imp = np.zeros(len(feature_names), dtype=float)
                name_to_idx = {name: i for i, name in enumerate(feature_names)}
                for k, v in score.items():
                    if k.startswith("f") and k[1:].isdigit():
                        idx = int(k[1:])
                        if 0 <= idx < len(imp):
                            imp[idx] = float(v)
                    elif k in name_to_idx:
                        imp[name_to_idx[k]] = float(v)
                if np.all(imp == 0):
                    imp = np.asarray(model.feature_importances_, dtype=float)
            return (
                pd.DataFrame({"feature": feature_names, "importance": imp})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
        except Exception:
            # fallback to sklearn wrapper feature_importances_
            imp = np.asarray(model.feature_importances_, dtype=float)
            return (
                pd.DataFrame({"feature": feature_names, "importance": imp})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

    # Linear models: abs(coef)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2 and coef.shape[0] == 1:
            coef = coef[0]
        imp = np.abs(np.ravel(coef))
        return (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # Tree models: feature_importances_
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        return (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # Fallback: permutation importance (subsample for speed)
    X_pi = X_te
    y_pi = y_te
    if len(X_pi) > 5000:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X_pi), size=5000, replace=False)
        X_pi = X_pi.iloc[idx]
        y_pi = y_pi.iloc[idx]

    pi = permutation_importance(
        model,
        X_pi,
        y_pi,
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return (
        pd.DataFrame({"feature": feature_names, "importance": pi.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_experiment(X_train, X_test, y_train, y_test, cols, feature_type, model_name, verbose=True, return_preds=False):
    """Train model on given columns and return test AUC (and optionally predictions + model)."""
    cols = [c for c in cols if c in X_train.columns and c in X_test.columns]
    if not cols:
        if verbose:
            print(f"  {model_name} / {feature_type}: AUC=None (no matching columns)", file=sys.stderr)
        return None
    X_tr = X_train[cols].fillna(0)
    X_te = X_test[cols].fillna(0)
    y_tr = y_train
    y_te = y_test
    if y_te.nunique() < 2:
        if verbose:
            print(f"  {model_name} / {feature_type}: AUC=None (only one class in test set)", file=sys.stderr)
        return None
    weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_tr)
    try:
        if model_name == 'lr':
            model, _ = run_lr(X_tr, y_tr, sample_weight=weights)
            y_prob = model.predict_proba(X_te)[:, 1]
        elif model_name == 'mlp':
            model, _ = run_mlp(X_tr, y_tr, sample_weight=weights)
            y_prob = model.predict_proba(X_te)[:, 1]
        elif model_name == 'xgb':
            model, _ = run_xgb(X_tr, y_tr, weights)
            y_prob = model.predict_proba(X_te)[:, 1]
        else:
            return None
        auc = float(roc_auc_score(y_te, y_prob))
        if return_preds:
            return auc, y_te, y_prob, model, cols, X_te
        return auc
    except Exception as e:
        if verbose:
            print(f"  {model_name} / {feature_type}: AUC=None (error: {e})", file=sys.stderr)
        return None


def main():
    # Reproducibility: fix RNGs before any splits or model fits
    np.random.seed(RANDOM_STATE)

    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    print("Loading data...")
    base = pd.read_csv(DATA_PATH)

    # Drop rows with no notes: all note-related columns are NaN
    note_cols = [c for c in base.columns if c.startswith("cn_") or c.startswith("nn_") or c in ["Notes", "TabularText", "Note"]]
    if note_cols:
        before = len(base)
        base = base[base[note_cols].notna().any(axis=1)]
        after = len(base)
        print(f"Filtered rows with no notes: kept {after} of {before} rows")
    else:
        print("Warning: no note columns found to filter on", file=sys.stderr)

    if OUTCOME_COL not in base.columns:
        print(f"Outcome column '{OUTCOME_COL}' not found.", file=sys.stderr)
        sys.exit(1)
    # y = base[OUTCOME_COL]

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    # )
    base_train = base.loc[base['OrderedDateKey'] < 20250101]
    base_test = base.loc[base['OrderedDateKey'] >= 20250101]

    drop_cols = [c for c in DROP_COLS if c in base.columns]
    X_train = base_train.drop(columns=drop_cols)
    X_test = base_test.drop(columns=drop_cols)
    y_train = base_train[OUTCOME_COL]
    y_test = base_test[OUTCOME_COL]
    print(f"Train {len(X_train)}, Test {len(X_test)}, positive rate train {y_train.mean():.4f} test {y_test.mean():.4f}")

    col_groups = get_feature_columns(X_train)
    print("Feature column counts:", {k: len(v) for k, v in col_groups.items()})
    if sum(len(v) for v in col_groups.values()) == 0:
        print("No feature columns found. Sample column names:", list(X_train.columns[:30]), file=sys.stderr)
    combos = get_feature_combinations(col_groups)
    # models = ['lr', 'mlp', 'xgb']
    models = ['xgb']

    results = []
    out_dir = "results";
    fi_dir = os.path.join(out_dir, "feature_importances")
    os.makedirs(fi_dir, exist_ok=True)
    for cols, feature_type in combos:
        for model_name in models:
            res = run_experiment(X_train, X_test, y_train, y_test, cols, feature_type, model_name, return_preds=True)
            if res is None:
                auc = None
            else:
                auc, y_true_series, y_prob_vec, model, used_cols, X_te_used = res
                # Save per-patient predictions on the test cohort
                pred_df = pd.DataFrame(
                    {
                        "y_pred_proba": y_prob_vec,
                        "y_test": y_true_series.values,
                    }
                )
                pred_path = os.path.join(
                    out_dir, f"test_preds_{model_name}_{feature_type}.csv"
                )
                pred_df.to_csv(pred_path, index=False)

                # Save trained model for this model / feature set
                model_path = os.path.join(
                    out_dir, f"model_{model_name}_{feature_type}.joblib"
                )
                joblib.dump(model, model_path)

                # Save top feature importances
                try:
                    fi_df = _compute_feature_importance(
                        model=model,
                        model_name=model_name,
                        X_te=X_te_used,
                        y_te=y_true_series,
                        feature_names=used_cols,
                    )
                    fi_path = os.path.join(fi_dir, f"top_features_{model_name}_{feature_type}.csv")
                    fi_df.head(50).to_csv(fi_path, index=False)
                except Exception as e:
                    print(
                        f"  Warning: could not compute feature importance for {model_name}/{feature_type}: {e}",
                        file=sys.stderr,
                    )
            results.append({'model': model_name, 'feature_type': feature_type, 'AUC': auc})
            print(f"  {model_name} / {feature_type}: AUC={auc}")

    out_df = pd.DataFrame(results)
    summary_path = os.path.join(out_dir, 'model_battery_summary_aucs.csv')
    out_df.to_csv(summary_path, index=False)
    print(f"Summary written to {summary_path}")
    return summary_path


if __name__ == '__main__':
    main()
