import logging

from omegaconf import OmegaConf
from sklearn.model_selection import GridSearchCV

from src.config import config as c
from src.dl_model import train_dl_model
from src.models import get_model


def tune_model(model_name, X_train, y_train):
    """Train a model by name. DL models go through PyTorch, others through GridSearchCV."""
    model_cfg = c.outputs[model_name]

    if model_name == 'mlp':
        params = OmegaConf.to_container(model_cfg.params, resolve=True)
        logging.info(f'Training DL model: {model_name} with params {params}')
        model = train_dl_model(X_train, y_train, **params)
        return model, None

    params = OmegaConf.to_container(model_cfg.params, resolve=True)
    base_model = get_model(model_cfg.name)

    logging.info(f'Training model: {model_name}')

    search = GridSearchCV(
        base_model,
        params,
        cv=c.training.cv_folds,
        scoring=c.training.scoring,
        n_jobs=1
    )
    search.fit(X_train, y_train)

    logging.info('Best model:\n%s', search.best_estimator_)
    logging.info('Best params: %s', search.best_params_)
    logging.info('Best cross-val metric: %.4f', search.best_score_)

    return search.best_estimator_, search.best_score_
