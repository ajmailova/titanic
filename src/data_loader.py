import os

import pandas as pd

from src.config import config as c


def load_data():
    """Read train and test CSVs from the paths in the config."""
    if os.path.exists(c.paths.train_file):
        train_df = pd.read_csv(c.paths.train_file)
        test_df = pd.read_csv(c.paths.test_file)
        return train_df, test_df
    else:
        print('Something is wrong')
