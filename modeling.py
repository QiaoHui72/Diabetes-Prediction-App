"""Reproducible tuning: preprocessing is fitted inside each training fold."""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FEATURES = ['Glucose', 'BloodPressure', 'BMI', 'Age']
ZERO_MEANS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def clean(data):
    data = data.copy()
    for col in ZERO_MEANS_MISSING:
        if col in data:
            data[col] = data[col].replace(0, np.nan)
    return data

def train_models(data, target='Outcome'):
    X = clean(data)[FEATURES]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Original search spaces are retained as a baseline, evaluated on identical folds.
    specs = {
        'Logistic Regression': (LogisticRegression(max_iter=2000),
            {'C': [.01, .1, 1, 10]}, {'C': [.001, .01, .1, 1, 10, 100], 'class_weight': [None, 'balanced']}),
        'K-Nearest Neighbors': (KNeighborsClassifier(),
            {'n_neighbors': [3,5,7,9,11,15,21,25], 'weights':['uniform','distance']},
            {'n_neighbors': [3,5,7,9,11,15,21,25,31,41], 'weights':['uniform','distance'], 'p':[1,2]}),
        'Random Forest': (RandomForestClassifier(random_state=42),
            {'n_estimators':[100,200], 'max_depth':[None,4,6,8], 'min_samples_leaf':[1,3,5]},
            {'n_estimators':[100,200], 'max_depth':[None,4,6,8], 'min_samples_leaf':[1,3,5,10], 'max_features':['sqrt',1.0]}),
        'Decision Tree': (DecisionTreeClassifier(random_state=42),
            {'max_depth':[3,4,5,6,8], 'min_samples_leaf':[1,5,10], 'criterion':['gini','entropy']},
            {'max_depth':[2,3,4,5,6,8], 'min_samples_leaf':[1,5,10,20], 'criterion':['gini','entropy'], 'ccp_alpha':[0,.005,.01]}),
        'Support Vector Machine': (SVC(probability=True, random_state=42), None,
            {'C':[.1,1,3,10,30], 'gamma':['scale',.01,.1,1], 'class_weight':[None,'balanced']}),
        'Gradient Boosting': (HistGradientBoostingClassifier(random_state=42), None,
            {'learning_rate':[.03,.1], 'max_iter':[100,200], 'max_leaf_nodes':[7,15], 'l2_regularization':[0,1,10]}),
    }
    results = {}
    # Only training folds are used for choosing models or settings.
    for name, (model, old_grid, grid) in specs.items():
        pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler',StandardScaler()), ('model',model)])
        baseline = None
        if old_grid:
            baseline = GridSearchCV(clone(pipe), {f'model__{k}':v for k,v in old_grid.items()}, cv=cv, scoring='accuracy', n_jobs=2).fit(X_train,y_train)
        params = {f'model__{k}':v for k,v in grid.items()}
        if name in ['K-Nearest Neighbors','Logistic Regression','Support Vector Machine']:
            params['scaler'] = [StandardScaler(), RobustScaler()]
        search = GridSearchCV(pipe, params, cv=cv, scoring='accuracy', n_jobs=2).fit(X_train,y_train)
        chosen = baseline if baseline is not None and baseline.best_score_ > search.best_score_ else search
        results[name] = {'pipeline':chosen.best_estimator_, 'cv_accuracy':float(chosen.best_score_),
            'cv_std':float(chosen.cv_results_['std_test_score'][chosen.best_index_]),
            'baseline_cv':float(baseline.best_score_) if baseline else None,
            'best_params':{k:str(v) for k,v in chosen.best_params_.items()}}
    best = max(results, key=lambda n: results[n]['cv_accuracy'])
    # Evaluate only after all choices are fixed. Never select using these test scores.
    for r in results.values():
        pred = r['pipeline'].predict(X_test)
        prob = r['pipeline'].predict_proba(X_test)[:,1]
        r.update(test_accuracy=float(accuracy_score(y_test,pred)), confusion=confusion_matrix(y_test,pred,labels=[0,1]),
                 precision=float(precision_score(y_test,pred,zero_division=0)), recall=float(recall_score(y_test,pred)),
                 f1=float(f1_score(y_test,pred)), auc=float(roc_auc_score(y_test,prob)), roc=roc_curve(y_test,prob)[:2])
    return results, best, FEATURES.copy()
