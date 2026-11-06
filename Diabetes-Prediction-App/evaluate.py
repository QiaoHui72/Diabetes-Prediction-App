"""Optional standalone evaluation: python evaluate.py"""
from pathlib import Path
import pandas as pd
from modeling import train_models

if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    results, best, _ = train_models(pd.read_csv(folder / 'diabetes.csv'))
    table = pd.DataFrame([{
        'Model': name, 'Original search CV (%)': r['baseline_cv'] * 100 if r['baseline_cv'] is not None else None,
        'Tuned CV (%)': r['cv_accuracy'] * 100, 'Test accuracy (%)': r['test_accuracy'] * 100,
        'Recall (%)': r['recall'] * 100, 'Precision (%)': r['precision'] * 100,
        'F1': r['f1'], 'ROC AUC': r['auc'],
    } for name, r in results.items()])
    table.round(3).to_csv(folder / 'evaluation_results.csv', index=False)
    print(table.to_string(index=False))
    print('Selected using CV:', best)
