# Titanic: Passenger Survival Prediction

A small educational project based on the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic). The goal is to predict which passengers survived the shipwreck using classic ML techniques and a simple neural network.

## Results

| Metric                             | Value       |
|------------------------------------|-------------|
| Cross-validation (5-fold accuracy) | **0.8462**  |
| Public leaderboard                 | **0.78468** |

The best result came from a soft-voting ensemble of XGBoost, LightGBM, and Logistic Regression with weights `[1, 1, 2]`. Logistic regression got the highest weight - on this small dataset it turned out more stable than the boosting models, which tend to overfit on 891 samples.

A two-layer PyTorch MLP was also trained as an additional experiment. On its own it scores slightly lower than the ensemble, but it gives a useful baseline for comparing neural approaches with classical models.

## How It Works

**Features.** From the original 11 columns I built 15: title from the name (`Initial`), family size and type, a flag for traveling alone, vowel and consonant counts in the name, ticket and surname frequencies, deck from the cabin number, and grouped age/fare intervals.

**Preprocessing.** Missing ages are filled by median within `Sex + Pclass` groups. Categorical features are one-hot encoded, ordinal ones (`Pclass`, `family_type`) go through `OrdinalEncoder` with an explicit order, and numeric ones are scaled. All statistics are computed on train only - no leakage.

**Models.** Logistic Regression, Ridge, Lasso, KNN, Random Forest, XGBoost, LightGBM, CatBoost, plus Voting and Stacking ensembles. Hyperparameters are stored in `config.py` and tuned with `GridSearchCV`.

**Neural network.** A small MLP built on PyTorch (`src/dl_model.py`) with configurable hidden layers, batch normalization, dropout, and activation. Architecture and training parameters live in `config.py` under the `mlp` block, so trying a different setup only means editing a few lines there.

## Structure

```
titanic/
├── data/
├── src/
│   ├── config.py         # all settings in one place
│   ├── data_loader.py
│   ├── preprocessing.py  # feature engineering and ColumnTransformer
│   ├── models.py         # model factory
│   ├── dl_model.py       # PyTorch MLP and training loop
│   ├── trainer.py        # training and hyperparameter search
│   ├── evaluator.py      # submission and model saving
│   └── utils.py          # logging
├── notebooks/EDA.ipynb   # notebook with analysis
├── outputs/              # created automatically
├── main.py
├── requirements.txt
└── README.md
```

## Running

```bash
git clone <url>
cd titanic
pip install -r requirements.txt
python main.py
```

Before running, download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/c/titanic/data) and drop them into `data/`. The dataset is not included in the repo due to Kaggle's terms.

After the run you'll find:

- `outputs/submissions/submission_<model>_<timestamp>.csv` - ready for Kaggle,
- `outputs/models/<model>_<timestamp>.pkl` - the trained sklearn model,
- `outputs/models/<model>_<timestamp>.pt` - the trained PyTorch model,
- `outputs/logs/training.log` - training log.

## Switching Models

Everything lives in `src/config.py`. To try a different model, just change one line in `main.py`:

```python
model_name = 'xgboost'
```

Other options: `logreg`, `knn`, `random_forest`, `lightgbm`, `catboost`, `voting_xgb_lgbm_rf`, `stacking_ridge`, and `mlp` for the neural network. Each model's hyperparameters are defined in the same config file.