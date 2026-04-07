from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.utils import class_weight
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
#import torch
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import os

# For stacking
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.ensemble         import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network   import MLPClassifier
from sklearn.svm              import SVC

from sklearn.ensemble     import GradientBoostingClassifier
from sklearn.ensemble     import StackingClassifier

# GPU set up
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
GPU_ID = 0

RANDOM_STATE = 14

# Fixed XGB params for risk-factor pipeline (sparse, many patients, low positive rate)
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'enable_categorical': True,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.01,
    'verbosity': 0
}

def run_xgb(X_train, y_train, weights):
   
    xgb_param_grid = {
        'n_estimators': Integer(100, 2000),
        'max_depth': Integer(0, 50),
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
        'min_child_weight': Integer(0, 10),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0)
    }

    # split on patients for cross-val
    kf = KFold(n_splits=3, shuffle=True, random_state=42).split(X_train, y_train)

    # get class weights
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )

    xgb = XGBClassifier(objective='binary:logistic', tree_method='hist', enable_categorical=True, n_jobs=1, nthread=1, device='cuda')
    #xgb_grid_search = GridSearchCV(xgb, xgb_param_grid, cv=kf, scoring='roc_auc', verbose=10)
    #xgb_grid_search = BayesSearchCV(xgb, xgb_param_grid, n_iter=30, cv=kf, scoring='roc_auc_ovr', verbose=10, n_jobs=1)
    xgb_grid_search = BayesSearchCV(xgb, xgb_param_grid, n_iter=10, cv=kf, scoring='roc_auc', verbose=10, n_jobs=1)
    xgb_grid_search.fit(X_train, y_train, sample_weight=weights)

    return xgb_grid_search.best_estimator_, xgb_grid_search.best_params_

def run_xgb_plain(X_train, y_train, weights):
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    
    xgb = XGBClassifier(objective='binary:logistic', tree_method='hist', device='cuda:0', enable_categorical=True, colsample_bytree=0.9513472132272754, learning_rate=0.016567942236629694, max_depth=26, min_child_weight=10, n_estimators=1258, subsample=0.7272877442083587, n_jobs=1, nthread=1)
    xgb.fit(X_train, y_train, sample_weight=weights)
    
    return xgb, {}


def run_tabnet(X_train, y_train):
    # get the categorical columns (hacky...)
    cat_idxs = []
    cat_dims = []
    for i, c in enumerate(X_train.columns):
        if (c.startswith('icd_') or c.startswith('rx_') or c.startswith('cmp_') or 
            c == 'AlteredMentalStatus' or c == 'HighFever' or c == 'LowFever' or 
            c == 'HighPulse' or c == 'LowOxygen'):
            cat_idxs.append(i)
            cat_dims.append(2)

    # Doing my own manual crossval to find the best epoch!
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_best_epochs = []
    fold_best_aucs = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        print(f"Training fold {fold + 1}/{kf.n_splits}")

        X_train_this, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_this, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        clf = TabNetClassifier(cat_idxs=cat_idxs, cat_dims=cat_dims)

        clf.fit(
            X_train_this.values, y_train_this.values,
            eval_set=[(X_val.values, y_val.values)],
            eval_metric=['auc'],
            max_epochs=1000,
            patience=20, # patience for early stopping
            weights=1 # makes it balanced
        )

        best_epoch = np.argmax(clf.history['val_0_auc'])
        best_auc = np.max(clf.history['val_0_auc'])
        fold_best_epochs.append(best_epoch)
        fold_best_aucs.append(best_auc)

    # average best epochs
    avg_best_epoch = np.mean(fold_best_epochs).astype(int)
    print(f"Average best epoch across folds: {avg_best_epoch}")
    print(f"Best AUCs per fold: {fold_best_aucs}")
    print(f"Mean AUC across folds: {np.mean(fold_best_aucs)}")

    # retrain model on full data
    clf_final = TabNetClassifier(cat_idxs=cat_idxs, cat_dims=cat_dims)
    clf_final.fit(X_train.values, y_train.values, max_epochs=avg_best_epoch, weights=1)

    return clf_final, {'max_epochs': avg_best_epoch}

def run_gp(X_train, y_train):
    gp = GaussianProcessClassifier()
    gp.fit(X_train, y_train)

    return gp, {}

def run_ada(X_train, y_train):   
    ada_param_grid = {
        'n_estimators': Integer(20, 100),
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
        'algorithm': ['SAMME', 'SAMME.R']
    }

    # split on patients for cross-val
    kf = KFold(n_splits=3, shuffle=True, random_state=42).split(X_train, y_train)

    # get class weights
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )

    ada = AdaBoostClassifier()
    ada_grid_search = BayesSearchCV(ada, ada_param_grid, n_iter=50, cv=kf, scoring='roc_auc', verbose=10, n_jobs=1)
    ada_grid_search.fit(X_train, y_train, sample_weight=classes_weights)

    return ada_grid_search.best_estimator_, ada_grid_search.best_params_
    
def run_stacker(X_train, y_train):    
    neighbors = KNeighborsClassifier()
    rf        = RandomForestClassifier()
    # Parameters pulled from an independent trial
    xgb       = XGBClassifier(objective='binary:logistic', tree_method = 'gpu_hist', enable_categorical=True, 
                              n_jobs=1, nthread=1, gpu_id=GPU_ID,
                              colsample_bytree = 0.7693967938433998, 
                              learning_rate = 0.013832395421659698, 
                              max_depth = 0, 
                              min_child_weight = 0, 
                              n_estimators  = 2000,
                              subsample = 0.7965758267489123)
    xgb._estimator_type = "classifier"
    gp        = GaussianProcessClassifier()
    ab        = AdaBoostClassifier(algorithm = "SAMME")
    mlp       = MLPClassifier(alpha=1, max_iter = 1000)

    estimators = [
        # ("KNeighbors", neighbors),
        ("RandomForest", rf),
        ("XGBoost", xgb),
        # ("Gaussian Process", gp),
        ("ADA Boost", ab),
        # ("MLPClassifier", mlp)
    ]

    # get class weights
    classes_weights  = class_weight.compute_sample_weight(
        class_weight ='balanced',
        y            = y_train
    )

    stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=GradientBoostingClassifier())
    stacking_classifier.fit(X_train, y_train, sample_weight=classes_weights)

    return stacking_classifier, {}


def run_lr(X_train, y_train, sample_weight=None):
    """Logistic regression tuned for sparse columns, many patients, low positive rate."""
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['saga'],
        'max_iter': [10000],
        'tol': [1e-3],
        'class_weight': ['balanced'],
        'random_state': [RANDOM_STATE],
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    lr = LogisticRegression()
    search = GridSearchCV(lr, param_grid, cv=kf, scoring='roc_auc', n_jobs=1, verbose=0)
    search.fit(X_train, y_train, sample_weight=sample_weight)
    return search.best_estimator_, search.best_params_


def run_mlp(X_train, y_train, sample_weight=None):
    """MLP tuned for sparse columns, many patients, low positive rate. (sample_weight ignored; uses class_weight.)"""
    param_grid = {
        'hidden_layer_sizes': [(64, 32), (100, 50), (128, 64)],
        'alpha': [0.01, 0.1, 0.5],
        'max_iter': [500],
        'early_stopping': [True],
        'class_weight': ['balanced'],
        'random_state': [RANDOM_STATE],
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    mlp = MLPClassifier()
    search = GridSearchCV(mlp, param_grid, cv=kf, scoring='roc_auc', n_jobs=1, verbose=0)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def run_xgb_fixed(X_train, y_train, sample_weight=None):
    """XGBoost with fixed params for risk-factor pipeline + CV on train."""
    xgb = XGBClassifier(**XGB_PARAMS, n_jobs=1)
    w = sample_weight
    if w is None:
        w = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

    # Cross-validated AUC on the training data
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        xgb,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        fit_params={"sample_weight": w},
    )

    # Fit final model on full training data
    xgb.fit(X_train, y_train, sample_weight=w)
    params = XGB_PARAMS.copy()
    params["cv_auc_mean"] = float(np.mean(cv_scores))
    params["cv_auc_std"] = float(np.std(cv_scores))
    return xgb, params

    