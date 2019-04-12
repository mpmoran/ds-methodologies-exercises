import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import scipy.stats as stats

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

pd.options.display.float_format = "{:20,.2f}".format

# TODO
# - [ ] drop basementsqft (has 43 non-null vals)
# - [ ] drop decktypeid (has 658 non-null vals)
# -

# include only single unit properties (e.g. no duplexes, no land/lot, ...)
# For some properties, you will need to use multiple fields to estimate
# whether it is a single unit property.


def single_unit(df: pd.DataFrame) -> pd.DataFrame:
    # unit cnt is 1 or nan and has at least one bathroom and bedroom and
    # square footage above 500

    df = df[(df.unitcnt == 1) | (df.unitcnt.isnull())]
    df = df[
        (df.bathroomcnt > 0)
        & (df.bedroomcnt > 0)
        & (df.calculatedfinishedsquarefeet > 500)
    ]
    df = df[
        (~df.propertylandusedesc.str.contains("Duplex"))
        & (~df.propertylandusedesc.str.contains("Triplex"))
    ]

    return df


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


def numeric_to_categorical(df: pd.DataFrame, cols: tuple) -> pd.DataFrame:
    to_coerce = {col: "category" for col in cols}
    return df.astype(to_coerce)


def zillow_num_to_cat(df: pd.DataFrame) -> pd.DataFrame:
    cols = (
        "buildingqualitytypeid",
        "fips",
        "rawcensustractandblock",
        "regionidcity",
        "regionidcounty",
        "regionidneighborhood",
        "regionidzip",
        "unitcnt",
        "censustractandblock",
    )
    return numeric_to_categorical(df, cols)


def prepare_zillow(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(single_unit)
        .pipe(zillow_num_to_cat)
        .pipe(zillow_drop_cols)
        .pipe(zillow_fill_cols_0)
        .pipe(zillow_impute_cols)
        .pipe(zillow_drop_rows)
    )


def df_missing_vals_by_col(df: pd.DataFrame) -> pd.DataFrame:
    null_count = df.isnull().sum()
    null_percentage = (null_count / df.shape[0]) * 100
    empty_count = pd.Series(
        ((df == " ") | (df == "") | (df == "nan") | (df == "NaN")).sum()
    )
    return pd.DataFrame(
        {
            "nmissing": null_count,
            "percentage": null_percentage,
            "nempty": empty_count,
        }
    )


def df_missing_vals_by_row(df: pd.DataFrame) -> pd.DataFrame:
    null_count = df.isnull().sum(axis=1)
    null_percentage = (null_count / df.shape[1]) * 100
    empty_count = pd.Series(((df == " ") | (df == "")).sum(axis=1))
    return pd.DataFrame(
        {
            "nmissing": null_count,
            "percentage": null_percentage,
            "nempty": empty_count,
        }
    )


def df_impute_0(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    return DataFrame with only the NaNs in cols replaced with 0
    """
    impute_vals = {col: 0 for col in cols}
    return df.fillna(value=impute_vals)


def zillow_fill_cols_0(df: pd.DataFrame) -> pd.DataFrame:
    return df_impute_0(df, ["hashottuborspa", "poolcnt", "pooltypeid7"])


def df_impute_col(df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
    imp = SimpleImputer(**kwargs)
    imp.fit(df[[col]].dropna())
    df_imputed = df.copy()
    df_imputed.loc[:, col] = imp.transform(df_imputed[[col]])

    return df_imputed


def zillow_impute_bldqlty(df: pd.DataFrame) -> pd.DataFrame:
    return df_impute_col(df, "buildingqualitytypeid", strategy="most_frequent")


def zillow_impute_prptyzoning(df: pd.DataFrame) -> pd.DataFrame:
    return df_impute_col(df, "propertyzoningdesc", strategy="most_frequent")


def zillow_impute_regionidcity(df: pd.DataFrame) -> pd.DataFrame:
    return df_impute_col(df, "regionidcity", strategy="most_frequent")


def zillow_impute_regionidzip(df: pd.DataFrame) -> pd.DataFrame:
    return df_impute_col(df, "regionidzip", strategy="most_frequent")


def zillow_impute_yearbuilt(df: pd.DataFrame) -> pd.DataFrame:
    return df_impute_col(df, "yearbuilt", strategy="most_frequent")


def zillow_impute_unitcnt(df: pd.DataFrame) -> pd.DataFrame:
    """
    impute 1 for unitcnt
    """
    return df_impute_col(df, "unitcnt", strategy="constant", fill_value=1)


def zillow_impute_taxdelinquencyflag(df: pd.DataFrame) -> pd.DataFrame:
    """
    impute N for taxdelinquencyflag
    """
    return df_impute_col(
        df, "taxdelinquencyflag", strategy="constant", fill_value="N"
    )


def zillow_impute_cols(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(zillow_fill_cols_0)
        .pipe(zillow_impute_bldqlty)
        .pipe(zillow_impute_prptyzoning)
        .pipe(zillow_impute_regionidcity)
        .pipe(zillow_impute_regionidzip)
        .pipe(zillow_impute_yearbuilt)
        .pipe(zillow_impute_land_sqft)
        .pipe(zillow_impute_taxdelinquencyflag)
        .pipe(zillow_impute_unitcnt)
    )


def zillow_drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        "airconditioningdesc",
        "finishedsquarefeet12",
        "structuretaxvaluedollarcnt",
        "basementsqft",
        "calculatedbathnbr",
        "decktypeid",
        "finishedfloor1squarefeet",
        "finishedsquarefeet13",
        "finishedsquarefeet15",
        "finishedsquarefeet50",
        "finishedsquarefeet6",
        "fireplacecnt",
        "fullbathcnt",
        "garagecarcnt",
        "garagetotalsqft",
        "hashottuborspa",
        "heatingorsystemdesc",
        "poolsizesum",
        "pooltypeid10",
        "pooltypeid2",
        "taxdelinquencyyear",
        "threequarterbathnbr",
        "yardbuildingsqft17",
        "yardbuildingsqft26",
        "numberofstories",
        "fireplaceflag",
        "architecturalstyledesc",
        "buildingclassdesc",
        "storydesc",
        "typeconstructiondesc",
        "regionidneighborhood",
    ]

    return df.drop(columns=cols_to_drop)


def zillow_impute_land_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    return DataFrame with missing values for lotsizesquarefeet filled in
    with results of a linear regression model using landtaxvaluedollarcnt
    as the independent variable
    """
    # df = df[df.landtaxvaluedollarcnt != np.nan] (necessary?)
    df_tmp = df[["landtaxvaluedollarcnt", "lotsizesquarefeet"]].dropna()

    lm = LinearRegression(fit_intercept=False)
    lm.fit(df_tmp[["landtaxvaluedollarcnt"]], df_tmp[["lotsizesquarefeet"]])

    pred = lm.predict(
        df[df.lotsizesquarefeet.isna()][["landtaxvaluedollarcnt"]]
    )

    imputed = df.copy()
    imputed.loc[imputed.lotsizesquarefeet.isna(), "lotsizesquarefeet"] = pred

    return imputed


def zillow_drop_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "taxvaluedollarcnt",
        "landtaxvaluedollarcnt",
        "taxamount",
        "censustractandblock",
    ]
    return df.dropna(subset=cols)
