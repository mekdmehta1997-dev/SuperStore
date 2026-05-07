import pandas as pd

from src.app_config import DROP_COLUMNS

def load_data(filepath):

    df = pd.read_csv(filepath, encoding='latin1')

    df = df.drop(DROP_COLUMNS, axis=1)

    df.drop_duplicates(inplace=True)

    df['Profit'] = df['Profit'].abs()

    return df