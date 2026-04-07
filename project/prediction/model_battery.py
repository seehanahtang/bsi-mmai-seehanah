import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, f1_score
from sklearn.utils import class_weight
from datetime import datetime
import time
import sys
import gc
import os
import psutil
import signal
import re
from models import *


# set comp limits
p = psutil.Process(os.getpid())
p.cpu_affinity([0])
os.nice(19)

def handle_sigterm(signum, frame): print("Ignoring sigterm")
signal.signal(signal.SIGTERM, handle_sigterm)

# choose which model to run on
MODEL = sys.argv[1] # can be 'xgb' or 'tabnet'
OUTCOME = sys.argv[2] # choose between 'binary' and 'multiclass'
TRAIN_DATA = "HH_emergency_before_2025_summary_notes" # sys.argv[4] 
TEST_DATA = "HH_emergency_2025_summary_notes" # sys.argv[5] 

# TRAIN_DATA = "HH_emergency_before_2025" # sys.argv[4] 
# TEST_DATA = "HH_emergency_2025" # sys.argv[5] 

SEED  = 14
RESULTS_DIR = f'results/{datetime.now().strftime("%d-%m-%Y-%H:%M")}/'
LABELS_DIR = RESULTS_DIR + 'labels/'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    os.makedirs(LABELS_DIR)

# READING DATA
base_train = pd.read_csv(f'data/base_{TRAIN_DATA}.csv')
base_test = pd.read_csv(f'data/base_{TEST_DATA}.csv')
# base_train = pd.read_csv(f'/datafs_a/carolgao/haim14_data/blood_infection/base_{TRAIN_DATA}.csv')
# base_test = pd.read_csv(f'/datafs_a/carolgao/haim14_data/blood_infection/base_{TEST_DATA}.csv')
bese_train = base_train.drop_duplicates()
base_test = base_test.drop_duplicates()
base_train    = base_train.drop(columns = ['lab_lactic acid', 'lab_INR', 'lab_P/F ratio'])
base_test    = base_test.drop(columns = ['lab_lactic acid', 'lab_INR', 'lab_P/F ratio'])

# merge with risk factors 
risk = pd.read_csv(f"/datafs_a/carolgao/haim14_data/blood_infection/HH_risk_factors.csv")
base_train = pd.merge(base_train, risk, on = 'EncounterKey', how = 'left')
base_test = pd.merge(base_test, risk, on = 'EncounterKey', how = 'left')

X_train    = base_train.drop(columns = ['PatientDurableKey', 'EncounterKey', 'Positive', 'OrderedDateKey', 'OrderedTimeOfDay', 'FalsePositive'])
X_test    = base_test.drop(columns = ['PatientDurableKey', 'EncounterKey', 'Positive', 'OrderedDateKey', 'OrderedTimeOfDay', 'FalsePositive'])

if OUTCOME == "binary":
    y_train = base_train['Positive']
    y_test = base_test['Positive']
elif OUTCOME == "multiclass": 
    y_train = base_train['bsi_class']
    y_test = base_test['bsi_class']

print("Done preparing data\n\n")  

# TRAINS MODELS ON GIVEN COLUMNS AND PRINTS A REPORT

def model_report(cols, model_type = MODEL, title=None):   
    # get relevant columns
    X_train_this = X_train[cols]
    X_test_this = X_test[cols]
    
    # only keep patients with the selected features
    # X_train_this = X_train_this.dropna()
    # X_test_this = X_test_this.dropna()
    y_train_this = y_train[X_train_this.index]
    y_test_this = y_test[X_test_this.index]   

    base_train['uniform_weight'] = 1
    weights = base_train['uniform_weight']
    weights = weights[X_train_this.index]
    
    # Train the models
    start = time.time()

    match model_type:
        case 'xgb':
            print("Doing xgb")
            model, params = run_xgb(X_train_this, y_train_this, weights)
            y_prob = model.predict_proba(X_test_this)
            y_prob_train = model.predict_proba(X_train_this)
            y_pred = model.predict(X_test_this)
        
            importances = pd.Series(model.feature_importances_, index=X_train_this.columns).sort_values(ascending=False)
            
        case 'xgb_plain':
            print("Doing xgb without hyperparams")
            model, params = run_xgb_plain(X_train_this, y_train_this, weights)
            y_prob = model.predict_proba(X_test_this)
            y_prob_train = model.predict_proba(X_train_this)
            y_pred = model.predict(X_test_this)
        
            importances = pd.Series(model.feature_importances_, index=X_train_this.columns).sort_values(ascending=False)  
        case 'tabnet':
            print("Doing tabnet")
            model, params = run_tabnet(X_train_this, y_train_this)

            # tabnet does not accept pandas dataframes, so have to pass as np array
            y_prob = model.predict_proba(X_test_this.values)
            y_pred = model.predict(X_test_this.values)
        
            importances = pd.Series(model.feature_importances_, index=X_train_this.columns).sort_values(ascending=False)
        case 'gp':
            print("Doing Gaussian Process")
            model, params = run_gp(X_train_this, y_train_this)

            y_prob = model.predict_proba(X_test_this)
            y_pred = model.predict(X_test_this)

            importances = None

        case 'ada':
            print("Doing ADA Boost")
            model, params = run_ada(X_train_this, y_train_this)

            y_prob = model.predict_proba(X_test_this)
            y_pred = model.predict(X_test_this)

            importances = None
            
        case 'stacker':
            print("Doing Stacker")
            model, params = run_stacker(X_train_this, y_train_this)

            y_prob = model.predict_proba(X_test_this)
            y_prob_train = model.predict_proba(X_train_this)
            y_pred = model.predict(X_test_this)
        
            #importances = None
        case _:
            raise Exception("Invalid model type")

    elapsed = time.time() - start
    
    # Print out the report
    
    original_stdout = sys.stdout
    file = open(f'{RESULTS_DIR}{title}.txt', 'w')
    sys.stdout = file

    print(f'# {title if title else "Experiment Report"} #')
    print(f'Report generated at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
    print('\n\n')

    print(f'## Metadata')
    print(f'Trained on {TRAIN_DATA}')
    print(f'Tested on {TEST_DATA}')
    print(f'Outcome {OUTCOME}')
    print(f'Total time elapsed: {elapsed : .2f} sec')
    print(f'Model type: {model_type}')
    print(f'Number of columns: {len(cols)}')
    print(f'Train rows: {len(X_train_this)}')
    print(f'Test rows: {len(X_test_this)}')
    print(f'Total rows: {len(X_train_this) + len(X_test_this)}')
    print(f'Params: {params}')
    print('\n\n')

    print(f'## Metrics')
    if OUTCOME == "binary":
        print(f'F1: {f1_score(y_test_this, y_pred, average="weighted"):.4f}')
        print(f'AUC: {roc_auc_score(y_test_this, y_prob[:,1]):.4f}')
    elif OUTCOME == "multiclass": 
        print(f'F1: {f1_score(y_test, y_pred, average="weighted"):.4f}')
        print(f'AUC: {roc_auc_score(y_test, y_prob, multi_class = "ovr"):.4f}')
    
    print(f'Confusion matrix:\n{confusion_matrix(y_test_this, y_pred)}')

    print(f'Classification report:\n{classification_report(y_test_this, y_pred)}')

    print('\n\n')

    
    if importances is not None:
        importances.to_csv(f'{LABELS_DIR}/{title}_feature_importances.csv', header=True)
    #print(f'## Importances')
    #print(f'\nTop:\n{importances.head(20)}')
    #print(f'\nBottom:\n{importances.tail(20)}')
    
        
    file.close()  
    sys.stdout = original_stdout
       
    # Save predicted probabilities
    np.save(f'{LABELS_DIR}{title}_y_prob_test', y_prob)
    np.save(f'{LABELS_DIR}{title}_y_prob_train', y_prob_train)

    # Save true
    np.save(f'{LABELS_DIR}{title}_y_true_test', y_test_this)
    np.save(f'{LABELS_DIR}{title}_y_true_train', y_train_this)
    
    if OUTCOME == "binary": 
        result = [roc_auc_score(y_test_this, y_prob[:,1]), elapsed]
    elif OUTCOME == "multiclass": 
        result = [roc_auc_score(y_test, y_prob, multi_class = "ovr"), elapsed]

    # memory management

    del X_train_this
    del X_test_this
    del y_train_this
    del y_test_this
    del model
    del y_prob
    del y_pred
    del importances
    gc.collect()

    return result

# Isolate ICD columns
icd_columns   = [c for c in base_train.columns if (c.startswith('icd_'))]
# Isolate ICD chapter columns 
chapter_columns = [c for c in base_train.columns if (c.startswith('chapter_'))]
# Isolate complaints columns
comp_columns   = [c for c in base_train.columns if (c.startswith('cmp_'))]
# Isolate medication columns 
med_columns   = [c for c in base_train.columns if (c.startswith('rx_'))]
# Isolate demographics 
demo_columns = [c for c in base_train.columns if (c.startswith('demo_'))]
# Isolate vitals
vitals_columns = list(base_train.columns[(base_train.columns.isin(['Temp', 'SpO2', 'Pulse','Resp','HighFever','LowFever','HighPulse', 'LowOxygen']))])
# Isolate notes columns 
notes_columns   = [c for c in base_train.columns if (c.startswith('cn_') )]
# Tabtext columns 
tt_columns   = [c for c in base_train.columns if (c.startswith('tt_'))]
# Keyword columns 
kw_columns = [c for c in base_train.columns if (c.startswith('keyword_'))]
# Lab columns
lab_columns = [c for c in base_train.columns if (c.startswith('lab_'))]
# risk columns
risk_columns = [c for c in base_train.columns if (c.startswith('risk_'))]

EXPERIMENTS = [
    # [icd_columns, 'ICD'],
    # [vitals_columns, 'Vitals'],
    # [comp_columns, 'Comps'],
    # [med_columns, 'Meds'],
    # [notes_columns, 'Notes'],
    # [kw_columns, 'Keywords'],
    # [lab_columns, 'Labs'],
    # [risk_columns, 'Risks'],
    # [vitals_columns+comp_columns+med_columns+notes_columns+kw_columns+lab_columns+risk_columns, 'No ICD'], # TODO: check if we need keywork columns
    # [icd_columns+comp_columns+med_columns+notes_columns+kw_columns+lab_columns+risk_columns, 'No Vitals'],
    # [icd_columns+vitals_columns+med_columns+notes_columns+kw_columns+lab_columns+risk_columns, 'No Comps'],     
    [icd_columns+vitals_columns+comp_columns+notes_columns+kw_columns+lab_columns+risk_columns, 'No Meds'],
    # [icd_columns+vitals_columns+comp_columns+med_columns+kw_columns+lab_columns+risk_columns, 'No Notes'],    
    [icd_columns+vitals_columns+comp_columns+med_columns+notes_columns+lab_columns+risk_columns, 'No Keywords'],    
    # [icd_columns+vitals_columns+comp_columns+med_columns+notes_columns+kw_columns+risk_columns, 'No Labs'], 
    # [icd_columns+vitals_columns+comp_columns+med_columns+notes_columns+kw_columns+lab_columns, 'No risks'],
    [icd_columns+vitals_columns+comp_columns+notes_columns+lab_columns+risk_columns, 'no_keywords_meds'],    
    # [icd_columns+vitals_columns+comp_columns+med_columns+notes_columns+kw_columns+lab_columns+risk_columns, 'all'], # everything!
]

start = time.time()
for cols, title in EXPERIMENTS:
    print(f"Starting experiment: {title}\n")
    result = model_report(cols, MODEL, title)
    print(f"\nDone experiment: {title} in {result[1] : .2f} sec with metric {result[0] : .4f} \n\n")


print(f"\n\nDONE ALL EXPERIMENTS. In {time.time() - start : .2f} sec")

