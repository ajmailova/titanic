import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder

from src.config import config as c


def _extract_title(df):
    """Pull the title from Name and collapse rare ones into 'Other'."""
    df['Initial'] = df['Name'].str.extract(r'([A-Za-z]+)\.')
    df['Initial'] = df['Initial'].replace(
        ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer',
         'Col', 'Rev', 'Capt', 'Sir', 'Don', 'Dona'],
        ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other',
         'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Other']
    )
    return df


def _extract_surname(df):
    """Take the surname — everything before the first comma in Name."""
    df['Family'] = (df['Name'].str.split(',').str[0]
                    .str.replace(r'[^\w\s]', '', regex=True).str.strip())
    return df


def _add_new_features(df):
    """Build family, name, and deck features."""
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    df['alone'] = (df['family_size'] == 1).astype(int)
    df['vowel_count'] = df['Name'].str.count(r'[aeiouAEIOU]')
    df['consonant_count'] = df['Name'].str.count(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]')
    df['is_married'] = (df['Initial'] == 'Mrs').astype(int)

    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'M')
    df.loc[df['Deck'] == 'T', 'Deck'] = 'A'
    df['Deck'] = df['Deck'].replace(['A', 'B', 'C'], 'ABC')
    df['Deck'] = df['Deck'].replace(['D', 'E'], 'DE')
    df['Deck'] = df['Deck'].replace(['F', 'G'], 'FG')
    return df


def _categorize_family(size):
    if size == 1:
        return 'Alone'
    elif size in [2, 3, 4]:
        return 'Small'
    elif size in [5, 6]:
        return 'Medium'
    elif size in [7, 8, 11]:
        return 'Large'
    else:
        return 'Other'


def _add_family_type(df):
    df['family_type'] = df['family_size'].apply(_categorize_family)
    return df


def _calculate_consts(train):
    """Compute imputation constants from train only."""
    age_medians = train.groupby(['Sex', 'Pclass'])['Age'].median()
    fare_median = train['Fare'].median()
    embarked_mode = train['Embarked'].mode()[0]
    ticket_freq = train['Ticket'].value_counts().to_dict()
    fam_freq = train['Family'].value_counts().to_dict()
    return age_medians, fare_median, embarked_mode, fam_freq, ticket_freq


def _add_frequencies(df, fam_freq, ticket_freq):
    df['ticket_freq'] = df['Ticket'].map(ticket_freq).fillna(0).astype(int)
    df['fam_freq'] = df['Family'].map(fam_freq).fillna(0).astype(int)
    return df


def _impute_values(df, age_medians, fare_median, embarked_mode):
    """Fill missing Age, Fare, and Embarked using train statistics."""
    df['Age'] = df['Age'].fillna(
        df[['Sex', 'Pclass']].apply(tuple, axis=1).map(age_medians)
    )
    df['Age'] = df['Age'].fillna(age_medians.median())
    df['Fare'] = df['Fare'].fillna(fare_median)
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    return df


def _add_age_fare_features(df):
    """Bin Age and Fare into coarse intervals."""
    df['age_band'] = pd.cut(df['Age'],
                            bins=[-np.inf, 16, 32, 48, 64, np.inf],
                            labels=False).astype(int)
    df['fare_cat'] = pd.cut(df['Fare'],
                            bins=[-0.001, 7.91, 14.454, 31, 513],
                            labels=False).astype(int)
    return df


def build_preprocessor():
    """Wire up one-hot, ordinal, and numeric pipelines."""
    cat_columns = list(c.preprocessing.cat_columns)
    num_columns = list(c.preprocessing.num_columns)
    ord_columns = list(c.preprocessing.ord_columns)

    ohe_pipe = Pipeline([
        ('ohe', OneHotEncoder(drop=c.preprocessing.one_hot_drop,
                              handle_unknown='ignore',
                              sparse_output=False))
    ])
    ord_pipe = Pipeline([
        ('ord', OrdinalEncoder(
            categories=[
                [1, 2, 3],
                ['Alone', 'Small', 'Medium', 'Large', 'Other']
            ],
            handle_unknown='use_encoded_value',
            unknown_value=-1))
    ])
    num_pipe = Pipeline([
        ('num', MinMaxScaler())
    ])

    return ColumnTransformer([
        ('ohe', ohe_pipe, cat_columns),
        ('ord', ord_pipe, ord_columns),
        ('num', num_pipe, num_columns)
    ], remainder='passthrough').set_output(transform='pandas')


def preprocessing_data(train, test):
    """Run the full feature pipeline and return processed train/test splits."""
    train_df = _extract_title(train)
    test_df = _extract_title(test)
    train_df = _extract_surname(train_df)
    test_df = _extract_surname(test_df)

    train_df = _add_new_features(train_df)
    test_df = _add_new_features(test_df)

    age_medians, fare_median, embarked_mode, fam_freq, ticket_freq = _calculate_consts(train_df)

    train_df = _impute_values(train_df, age_medians, fare_median, embarked_mode)
    test_df = _impute_values(test_df, age_medians, fare_median, embarked_mode)

    train_df = _add_frequencies(train_df, fam_freq, ticket_freq)
    test_df = _add_frequencies(test_df, fam_freq, ticket_freq)

    train_df = _add_family_type(train_df)
    test_df = _add_family_type(test_df)

    train_df = _add_age_fare_features(train_df)
    test_df = _add_age_fare_features(test_df)

    test_ids = test_df['PassengerId']

    train_df = train_df.drop(columns=c.preprocessing.drop_columns)
    test_df = test_df.drop(columns=c.preprocessing.drop_columns)

    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(test_df)

    return X_train_processed, y_train, X_test_processed, test_ids
