from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.config import config as c


def get_model(model_name, random_state=c.constants.random_state):
    """Build a model or ensemble by name."""
    xgb = XGBClassifier(random_state=random_state, colsample_bytree=0.6,
                        learning_rate=0.2, max_depth=3, n_estimators=150,
                        subsample=1.0, verbosity=0)
    ridge = RidgeClassifier(random_state=random_state, alpha=0.01)
    rf = RandomForestClassifier(random_state=random_state, min_samples_split=3,
                                class_weight=None, max_depth=7,
                                min_samples_leaf=1, n_estimators=100)
    lr = LogisticRegression(random_state=random_state, max_iter=4000, C=100.0)
    lgbm = LGBMClassifier(random_state=random_state, learning_rate=0.2,
                          max_depth=7, n_estimators=80, num_leaves=10,
                          verbose=-1)

    if model_name == 'LogisticRegression':
        return LogisticRegression(random_state=random_state)
    elif model_name == 'Ridge':
        return RidgeClassifier(random_state=random_state)
    elif model_name == 'Lasso':
        return LogisticRegression(l1_ratio=1, solver='saga',
                                  random_state=random_state)
    elif model_name == 'RandomForest':
        return rf
    elif model_name == 'KNeighborsClassifier':
        return KNeighborsClassifier()
    elif model_name == 'XGBClassifier':
        return xgb
    elif model_name == 'LGBMClassifier':
        return lgbm
    elif model_name == 'CatBoostClassifier':
        return CatBoostClassifier(random_state=random_state, verbose=0)

    elif model_name == 'Voting_XGB_LGBM_RF':
        return VotingClassifier(
            estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
            voting='soft'
        )
    elif model_name == 'Voting_XGB_LGBM_LR':
        return VotingClassifier(
            estimators=[('xgb', xgb), ('lgbm', lgbm), ('lr', lr)],
            voting='soft'
        )
    elif model_name == 'Stacking_LogReg':
        return StackingClassifier(
            estimators=[('xgb', xgb), ('ridge', ridge), ('rf', rf)],
            final_estimator=LogisticRegression(max_iter=4000,
                                               random_state=random_state),
            cv=5
        )
    elif model_name == 'Stacking_Ridge':
        return StackingClassifier(
            estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
            final_estimator=RidgeClassifier(random_state=random_state,
                                            alpha=0.01),
            cv=5
        )
    else:
        raise ValueError(f'Unknown model name: {model_name}')
