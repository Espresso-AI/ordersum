import json
import os
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split


def get_df(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return pd.DataFrame(data)


def cnndm_train_df(
        path: str,
        shuffle: bool = False,
        val_ratio: float = 0.1,
        random_state: Optional[int] = 42,
):
    df1 = get_df(os.path.join(path, 'train.json'))
    df2 = get_df(os.path.join(path, 'valid.json'))

    df1 = df1.dropna()
    df2 = df2.dropna()
    df1 = df1.drop_duplicates(subset=['id'], ignore_index=True)
    df2 = df2.drop_duplicates(subset=['id'], ignore_index=True)

    if not shuffle:
        return df1, df2

    df = pd.concat([df1, df2], ignore_index=True)
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=shuffle)

    return train_df, val_df


def cnndm_test_df(path: str):
    df = get_df(os.path.join(path, 'test.json'))
    df = df.dropna()
    df = df.drop_duplicates(subset=['id'], ignore_index=True)
    return df
