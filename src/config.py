from omegaconf import OmegaConf

config = {
    'paths': {
        'data_dir': 'data',
        'train_file': 'data/train.csv',
        'test_file': 'data/test.csv',
        'submission_dir': 'outputs/submissions',
        'model_dir': 'outputs/models',
        'log_dir': 'outputs/logs'
    },
    'constants': {
        'random_state': 42,
        'test_size': 0.25
    },
    'preprocessing': {
        'drop_columns': ['Cabin', 'PassengerId', 'Name', 'Ticket', 'Family'],
        'cat_columns': ['Sex', 'Embarked', 'Initial', 'Deck'],
        'num_columns': ['SibSp', 'Parch', 'family_size', 'alone', 'vowel_count',
                        'consonant_count', 'Age', 'Fare', 'ticket_freq', 'fam_freq',
                        'age_band', 'fare_cat', 'is_married'],
        'ord_columns': ['Pclass', 'family_type'],
        'one_hot_drop': 'first'
    },
    'outputs': {
        'logreg': {
            'name': 'LogisticRegression',
            'params': {
                'C': [0.01, 0.1, 0.5, 1, 10, 100],
                'max_iter': [1000]
            }
        },
        'knn': {
            'name': 'KNeighborsClassifier',
            'params': {
                'n_neighbors': [2, 3, 5, 7, 9, 10, 15],
                'weights': ['uniform', 'distance']
            }
        },
        'ridge': {
            'name': 'Ridge',
            'params': {
                'alpha': [0.01, 0.015, 0.025, 0.5, 1, 10],
                'max_iter': [2000]
            }
        },
        'lasso': {
            'name': 'Lasso',
            'params': {
                'C': [0.01, 0.015, 0.025, 0.5, 1, 10],
                'max_iter': [2000]
            }
        },
        'random_forest': {
            'name': 'RandomForest',
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5, 7, 9, None],
                'min_samples_split': [3, 5, 6, 7, 10],
                'min_samples_leaf': [1, 2, 3, 4],
                'class_weight': [None, 'balanced']
            }
        },
        'xgboost': {
            'name': 'XGBClassifier',
            'params': {
                'n_estimators': [80, 100, 150],
                'learning_rate': [0.01, 0.09, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 7, 9, 10, None],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.1, 0.4, 0.6, 0.8, 1.0]
            }
        },
        'lightgbm': {
            'name': 'LGBMClassifier',
            'params': {
                'n_estimators': [80, 100, 120, 150, 170, 200],
                'learning_rate': [0.01, 0.15, 0.2, 0.25, 0.3, 0.5, 10],
                'max_depth': [2, 4, 6, 7, 9, 12, None],
                'num_leaves': [5, 10, 20, 30, 40]
            }
        },
        'catboost': {
            'name': 'CatBoostClassifier',
            'params': {
                'iterations': [80, 100, 150],
                'learning_rate': [0.15, 0.2, 0.3],
                'max_depth': [5, 6, 7, 10],
                'min_data_in_leaf': [5, 7, 10, 12]
            }
        },
        'voting_xgb_lgbm_rf': {
            'name': 'Voting_XGB_LGBM_RF',
            'params': {
                'voting': ['soft', 'hard'],
                'weights': [None, [1, 1, 1], [2, 1, 1], [1, 2, 1]]
            }
        },
        'voting_xgb_lgbm_lr': {
            'name': 'Voting_XGB_LGBM_LR',
            'params': {
                'voting': ['soft', 'hard'],
                'weights': [None, [1, 1, 1], [1, 1, 2]]
            }
        },
        'stacking_logreg': {
            'name': 'Stacking_LogReg',
            'params': {
                'final_estimator__C': [0.1, 1, 10]
            }
        },
        'stacking_ridge': {
            'name': 'Stacking_Ridge',
            'params': {
                'final_estimator__alpha': [0.1, 1, 10]
            }
        },
        'mlp': {
            'name': 'TorchMLP',
            'params': {
                'hidden_dims': (64, 32),
                'dropouts': (0.5, 0.3),
                'activation': 'relu',
                'optimizer_name': 'adam',
                'lr': 0.001,
                'epochs': 50,
                'batch_size': 32
            }
        }
    },
    'training': {
        'cv_folds': 5,
        'scoring': 'accuracy'
    }
}

config = OmegaConf.create(config)