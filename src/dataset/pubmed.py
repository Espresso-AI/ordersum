from .cnn_dm import *


def pubmed_train_df(
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

    df1 = df1[df1['text'].apply(len) >= 4]
    df2 = df2[df2['text'].apply(len) >= 4]

    if not shuffle:
        return df1, df2

    df = pd.concat([df1, df2], ignore_index=True)
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=shuffle)

    return train_df, val_df


def pubmed_test_df(path: str):
    df = get_df(os.path.join(path, 'test.json'))
    df = df.dropna()
    df = df.drop_duplicates(subset=['id'], ignore_index=True)
    df = df[df['text'].apply(len) >= 4]
    return df
