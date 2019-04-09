import pandas as pd

from sklearn.preprocessing import LabelEncoder

pd.options.display.float_format = "{:20,.2f}".format


def series_outliers(series: pd.Series, k: float = 1.5) -> pd.Series:
    """
    returns Series containing outliers
    """
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    return series[(series < lower_bound) | (series > upper_bound)]


def remove_outliers(
    df: pd.DataFrame, cols: list, k: float = 1.5
) -> pd.DataFrame:
    """
    return DataFrame without outliers in cols using IQR method
    with k as the multiplier
    """
    df_outless = df.copy()
    for col in cols:
        outliers = series_outliers(df_outless[col], k)
        df_outless = df_outless[~df_outless[col].isin(outliers)]

    return df_outless


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    encoder = LabelEncoder()
    encoder.fit(df.gender)
    return df.assign(gender_encode=encoder.transform(df.gender))


def data_prep(df: pd.DataFrame) -> pd.DataFrame:
    return df.pipe(
        remove_outliers, cols=["age", "annual_income", "spending_score"]
    ).pipe(encode_gender)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    head_df = df.head()
    print(f"HEAD\n{head_df}", end="\n\n")

    tail_df = df.tail()
    print(f"TAIL\n{tail_df}", end="\n\n")

    shape_tuple = df.shape
    print(f"SHAPE: {shape_tuple}", end="\n\n")

    describe_df = df.describe()
    print(f"DESCRIPTION\n{describe_df}", end="\n\n")

    print(f"INFORMATION")
    df.info()

    print()

    for col in df.columns:
        n = df[col].unique().shape[0]
        col_bins = min(n, 10)
        print(f"{col}:")
        if df[col].dtype in ["int64", "float64"] and n > 10:
            print(df[col].value_counts(bins=col_bins, sort=False))
        else:
            print(df[col].value_counts())
        print("\n")


def debug():
    import acquire_mall

    df = acquire_mall.get_mallcustomer_data()
    df = data_prep(df)

    return df
