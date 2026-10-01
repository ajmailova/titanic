import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

from src.data_loader import load_data
from src.preprocessing import preprocessing_data
from src.trainer import tune_model
from src.evaluator import generate_submission, save_model
from src.utils import setup_logging, ensure_dirs

ensure_dirs()
setup_logging()

model_name = 'logreg'

train_data, test_data = load_data()
X_train, y_train, X_test, test_ids = preprocessing_data(train_data, test_data)

estimator, best_score = tune_model(model_name, X_train, y_train)

generate_submission(estimator, model_name, X_test, test_ids)
save_model(estimator, model_name)