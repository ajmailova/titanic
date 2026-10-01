import logging
import os
from datetime import datetime

import joblib
import pandas as pd
import torch

from src.config import config as c
from src.dl_model import predict_dl_model


def generate_submission(model, model_name, X_test, passenger_ids):
    """Predict on the test set and save a Kaggle-ready CSV."""
    if model_name == 'mlp':
        y_pred = predict_dl_model(model, X_test)
    else:
        y_pred = model.predict(X_test)

    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': y_pred
    })
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'submission_{model_name}_{timestamp}.csv'

    path = os.path.join(c.paths.submission_dir, filename)
    submission.to_csv(path, index=False)

    logging.info('submission.csv saved! Path: %s', path)
    return submission


def save_model(model, model_name):
    """Save sklearn model as .pkl or PyTorch model as .pt."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    if model_name == 'mlp':
        filename = f'{model_name}_{timestamp}.pt'
        path = os.path.join(c.paths.model_dir, filename)
        torch.save(model.state_dict(), path)
    else:
        filename = f'{model_name}_{timestamp}.pkl'
        path = os.path.join(c.paths.model_dir, filename)
        joblib.dump(model, path)

    logging.info('Model saved! Path: %s', path)
    return path
