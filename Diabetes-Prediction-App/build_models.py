"""Run locally after changing data or tuning; never run during web startup."""
from pathlib import Path
import hashlib
import pickle
import importlib.metadata
import pandas as pd
from modeling import train_models

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    source = root / 'diabetes.csv'
    payload = {
        'dataset_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'versions': {name: importlib.metadata.version(name) for name in ['scikit-learn','numpy','scipy','pandas']},
        'models': train_models(pd.read_csv(source)),
    }
    (root / 'trained_models.pkl').write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    print('Saved tuned models and evaluation results.')
