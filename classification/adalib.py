from typing import Dict, List, Tuple
import io

# Wrangling
import pandas as pd
import numpy as np

# Exploring
import scipy.stats as stats

# Visualizing
import matplotlib.pyplot as plt

# import seaborn as sns

# Modeling
# import statsmodels.api as sm

# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    # mean_absolute_error,
    median_absolute_error,
)
from sklearn.feature_selection import f_regression

# from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# TODO: move the dataframe class from model.ipynb and create a copy function
# that does the copying easily so i can create a new model but have the same
# train and test data; have the class have X_train, X_test, y_train,
# y_test attributes that have get methods that will perform the operation
# on the train_df to extract the data


def df_metadata(df: pd.DataFrame) -> Tuple[int, tuple, str]:
    """
    return size, shape, and info of df
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    return df.size, df.shape, buffer.getvalue()


def df_print_metadata(df: pd.DataFrame) -> None:
    """
    print metadata of df
    """
    size, shape, info = df_metadata(df)
    print("DATAFRAME METADATA")
    print(f"Size: {size}")
    print()
    print(f"Shape: {shape[0]} x {shape[1]}")
    print()
    print("Info:")
    print(info, end="")


def df_peek(df: pd.DataFrame, nrows: int = 5) -> pd.DataFrame:
    """
    return DataFrame containing nrows from head and tail of df
    """
    return df.head(nrows).append(df.tail(nrows))


def df_print_summary(df: pd.DataFrame, columns: List[str] = []) -> None:
    """
    columns is a sequence of columns whose IQR and range will be calculated.
    If columns is not specified, all columns will be summarized

    print describe(), iqr of columns, and range of columns
    """
    if not columns:
        columns = df.columns

    print("SUMMARY")
    description = df.describe()
    print("Description:")
    print(description)
    print()
    print("IQR:")
    # TODO: write function that will identify which columns are numeric
    # so we don't run iqr on non numeric columns and get an error
    # the function should print out which columns iqr was not called on
    # because they were not numeric
    for col in columns:
        print(f"\t{col}: {stats.iqr(df[col])}")
    print()
    print("Range:")
    for col in columns:
        print(f"\t{col}: {df[col].max() - df[col].min()}")


def series_is_whole_nums(series: pd.Series) -> bool:
    """
    return True if series is whole numbers
    """
    try:
        return (series % 1 == 0).all()
    except TypeError:
        return False


def df_float_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    return DataFrame with all columns that are whole numbers converted
    to int type
    """
    to_coerce = {}
    for col in df.columns:
        if series_is_whole_nums(df[col]):
            to_coerce[col] = int
    return df.astype(to_coerce)


def df_missing_vals(df: pd.DataFrame) -> pd.Series:
    """
    return Series containing the number of NaNs for each column that
    contains at least one
    """
    null_counts = df.isnull().sum()
    return null_counts[null_counts > 0]


def df_print_missing_vals(df: pd.DataFrame) -> None:
    """
    print the number of NaNs for each column that contains at least one
    """
    print("\nMissing Values:\n")
    series_missing_vals = df_missing_vals(df)
    if len(series_missing_vals):
        print(series_missing_vals)
    else:
        print("No missing values")


def df_percent_missing_vals(df: pd.DataFrame) -> pd.Series:
    """
    return Series containing the percent of NaNs of each column
    """
    return (df.isnull().sum() / df.shape[0]) * 100


def df_print_percent_missing_vals(df: pd.DataFrame) -> None:
    """
    print the percent of NaNs of each column
    """
    print(df_percent_missing_vals)


def df_remove_outliers_normal(
    df: pd.DataFrame, cols: List[str], zscore_limit: int
) -> pd.DataFrame:
    """
    WARNING: use for normal distributions
    return DataFrame without rows that contain a cell under any of cols with a
    z-score that exceeds zscore_limit
    """
    df_clean = pd.DataFrame()
    for col in cols:
        df_clean[col] = df[col][
            (np.abs(stats.zscore(df[col])) < zscore_limit).all(axis=1)
        ]

    return df_clean


def df_remove_outliers(
    df: pd.DataFrame, cols: List[str], bottom_top_percentile: float
) -> pd.DataFrame:
    """
    return DataFrame without rows that contain a cell under any of cols with
    a value that is in the bottom or top percentile as specified by
    bottom_top_percentile

    e.g., if bottom_top_percentile is 3, the function will remove
    rows with cells whose value is in the bottom or top 3 percentile
    """
    df_clean = df.copy()
    for col in cols:
        btmval = stats.scoreatpercentile(df[col], bottom_top_percentile)
        topval = stats.scoreatpercentile(df[col], 100 - bottom_top_percentile)
        df_clean[col] = df[(df[col] >= btmval) & (df[col] <= topval)]
    return df_clean


def df_r_and_p_values(
    X: pd.DataFrame, y: pd.Series
) -> Dict[str, Tuple[float, float]]:
    """
    return a dict containing Pearson's R and p-value for each column
    in X
    """
    return {col: stats.pearsonr(X[col], y) for col in X.columns}


def df_print_r_and_p_values(X: pd.DataFrame, y: pd.Series) -> None:
    """
    print the Pearson's R and p-value for each column in X
    """
    print("Pearson's R")
    for k, v in df_r_and_p_values(X, y).items():
        col = k
        r, p = v
        print(f"{col}:")
        print(
            f"\tPearson's R is {r:.3f} with a significance p-value "
            f"of {p:.3f}\n"
        )


def linreg_print_model(
    y_label: str, y_intercept: float, coefs: List[float], x_label: List[str]
) -> None:
    """
    print equation of a linear regression model
    """
    print(f"{y_label} = {y_intercept:.2f}", end="")
    for coef, label in coefs, x_label:
        print(f" +\n\t{coef:.3f} * {label}", end="")
    print()


def linreg_fit_and_predict(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, float, float, str]]:
    """
    perform linear regression on train data and predict on the training
    and test data

    return predictions of the train and test data
    """
    lm = LinearRegression()
    lm.fit(x_train, y_train)

    preds_train = lm.predict(x_train)
    preds_test = lm.predict(x_test)

    y_label = y_train.columns[0]
    y_intercept = lm.intercept_[0]
    m = lm.coef_[0][0]
    x_label = x_train.columns[0]

    return preds_train, preds_test, (y_label, y_intercept, m, x_label)


def evaluate_model_train(x, y, preds):
    y_label = y.columns[0]
    x_label = x.columns[0]

    print("Model Evaluation on TRAIN Data")
    meanse = mean_squared_error(y, preds)
    print(f"\tMSE: {meanse:.3f}")

    medianae = median_absolute_error(y, preds)
    print(f"\tMAE: {medianae:.3f}")

    r2 = r2_score(y, preds)
    print(
        f"\t{r2:.2%} of the variance in {y_label} can be explained "
        f"by {x_label}."
    )
    print()

    print("P-VALUE")
    f_vals, p_vals = f_regression(x, y)
    print(f"\tTrain: {p_vals[0]:.3}")
    print()


def evaluate_model_test(x, y, preds):
    y_label = y.columns[0]
    x_label = x.columns[0]

    print("Model Evaluation on TEST Data")
    meanse = mean_squared_error(y, preds)
    print(f"\tMSE: {meanse:.3f}")

    medianae = median_absolute_error(y, preds)
    print(f"\tMAE: {medianae:.3f}")

    r2 = r2_score(y, preds)
    print(
        f"\t{r2:.2%} of the variance in {y_label} can be explained "
        f"by {x_label}."
    )
    print()

    print("P-VALUE")
    f_vals, p_vals = f_regression(x, y)
    print(f"\tTest: {p_vals[0]:.3}")
    print()


def plot_residuals(y_test, preds_test):
    y_label = y_test.columns[0]
    plt.scatter(preds_test, preds_test - y_test, c="g", s=20)
    plt.hlines(y=0, xmin=preds_test.min(), xmax=preds_test.max())
    plt.title("Residual plot")
    plt.ylabel("Residuals")
    plt.xlabel(y_label)
    plt.show()


def linreg_model(x_train, y_train, x_test, y_test):
    lm, preds_train, preds_test = linreg_fit_and_predict(
        x_train, y_train, x_test, y_test
    )

    evaluate_model_train(x_train, y_train, preds_train)
    evaluate_model_test(x_test, y_test, preds_test)

    plot_residuals(y_test, preds_test)


def evaluate_multi_model_train(X, y, preds):
    y_label = y.columns[0]
    X_labels = X.columns

    print("Model Evaluation on TRAIN Data")
    meanse = mean_squared_error(y, preds)
    print(f"\tMSE: {meanse:.3f}")

    medianae = median_absolute_error(y, preds)
    print(f"\tMAE: {medianae:.3f}")

    r2 = r2_score(y, preds)
    print(
        f"\t{r2:.2%} of the variance in {y_label} can be explained "
        f"by {X_labels}."
    )
    print()

    print("P-VALUE")
    f_vals, p_vals = f_regression(X, y)
    print(f"\tTrain: {p_vals[0]:.3}")
    print()


def evaluate_multi_model_test(X, y, preds):
    y_label = y.columns[0]
    X_labels = X.columns

    print("Model Evaluation on TEST Data")
    meanse = mean_squared_error(y, preds)
    print(f"\tMSE: {meanse:.3f}")

    medianae = median_absolute_error(y, preds)
    print(f"\tMAE: {medianae:.3f}")

    r2 = r2_score(y, preds)
    print(
        f"\t{r2:.2%} of the variance in {y_label} can be explained "
        f"by {X_labels}."
    )
    print()

    print("P-VALUE")
    f_vals, p_vals = f_regression(X, y)
    print(f"\tTest: {p_vals[0]:.3}")
    print()


def multi_linreg_fit_and_evaluate(X_train, y_train, X_test, y_test):
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    y_label = y_train.columns[0]
    y_intercept = lm.intercept_[0]
    print("Multivariate:")
    print(f"{y_label} = ")
    print(f"{y_intercept:.3f}")
    for i, col in enumerate(X_train.columns):
        coefficient = lm.coef_[0][i]
        print(f"+ {coefficient:.3}*{col}")

    preds_train = lm.predict(X_train)
    evaluate_multi_model_train(X_train, y_train, preds_train)

    preds_test = lm.predict(X_test)
    evaluate_model_test(X_test, y_test, preds_test)

    plot_residuals(y_test, preds_test)


def normalize_cols(df_train, df_test, cols):
    df_train_norm = pd.DataFrame()
    for col in cols:
        minimum = df_train[col].min()
        maximum = df_train[col].max()
        df_train_norm[f"{col}_norm"] = (df_train[col] - minimum) / (
            maximum - minimum
        )

    df_test_norm = pd.DataFrame()
    for col in cols:
        minimum = df_train[col].min()  # use the min and max from the train set
        maximum = df_train[col].max()
        df_test_norm[f"{col}_norm"] = (df_test[col] - minimum) / (
            maximum - minimum
        )
    return df_train_norm, df_test_norm


def logreg_fit_and_predict(X_train, y_train, X_test, y_test, **kwargs):
    logreg = LogisticRegression(**kwargs)
    logreg.fit(X_train, y_train)

    preds_train = logreg.predict(X_train)
    preds_test = logreg.predict(X_test)

    return preds_train, preds_test, logreg


def logreg_evaluate_model(y_true, y_preds):

    print("Accuracy:", accuracy_score(y_true, y_preds))
    print()

    confmatrix = confusion_matrix(y_true, y_preds)
    df_confmatrix = pd.DataFrame(
        {"Pred -": confmatrix[:, 0], "Pred +": confmatrix[:, 1]}
    )
    df_confmatrix.rename({0: "Actual -", 1: "Actual +"}, inplace=True)
    print("Confusion matrix:")
    print(df_confmatrix)
    print()

    print("Classification report:")
    classrep = classification_report(y_true, y_preds)
    print(classrep)
    print()

    tp = confmatrix[1][1]
    fp = confmatrix[0][1]
    tn = confmatrix[0][0]
    fn = confmatrix[1][0]
    print("Rates:")
    print("True positive rate:", tp / (tp + fn))
    print("False positive rate:", fp / (fp + tp))
    print()
    print("True negative rate:", tn / (tn + fp))
    print("False negative rate:", fn / (fn + tn))
    print()


def logreg_model(X_train, y_train, X_test, y_test):
    preds_train, preds_test, _ = logreg_fit_and_predict(
        X_train, y_train, X_test, y_test
    )

    print("TRAIN EVALUATION")
    logreg_evaluate_model(y_train, preds_train)

    print("TEST EVALUATION")
    logreg_evaluate_model(y_test, preds_test)


def dectree_fit_and_predict(X_train, y_train, X_test, y_test, **kwargs):
    dectree = DecisionTreeClassifier(**kwargs)
    dectree.fit(X_train, y_train)

    preds_train = dectree.predict(X_train)
    preds_test = dectree.predict(X_test)

    return preds_train, preds_test, dectree.classes_, dectree


def dectree_evaluate_model(y_true, y_preds, classes):

    print("Accuracy:", accuracy_score(y_true, y_preds))
    print()

    confmatrix = confusion_matrix(y_true, y_preds)
    frame = {
        f"Pred {clas}": confmatrix[:, i] for i, clas in enumerate(classes)
    }
    df_confmatrix = pd.DataFrame(frame)
    for i, clas in enumerate(classes):
        df_confmatrix.rename({i: f"Actual {clas}"}, inplace=True)
    print("Confusion matrix:")
    print(df_confmatrix)
    print()

    print("Classification report:")
    classrep = classification_report(y_true, y_preds)
    print(classrep)
    print()

    for i, clas in enumerate(classes):
        total_row = confmatrix[i].sum()
        tru = confmatrix[i][i]

        total_col = confmatrix[:, i].sum()
        fls = total_col - tru

        print(f"True {clas} rate: {tru / total_row: .3f}")
        print(f"False {clas} rate: {fls / total_col: .3f}")
        print()


def dectree_model(X_train, y_train, X_test, y_test):
    preds_train, preds_test, classes, _ = dectree_fit_and_predict(
        X_train, y_train, X_test, y_test
    )

    print("TRAIN EVALUATION")
    dectree_evaluate_model(y_train, preds_train, classes)

    print("TEST EVALUATION")
    dectree_evaluate_model(y_test, preds_test, classes)


def knn_fit_and_predict(X_train, y_train, X_test, y_test, **kwargs):
    knn = KNeighborsClassifier(**kwargs)
    knn.fit(X_train, y_train)

    preds_train = knn.predict(X_train)
    preds_test = knn.predict(X_test)

    return preds_train, preds_test, knn.classes_, knn


# TODO: I may be able to combine this with dectree_evaluate_model
def knn_evaluate_model(y_true, y_preds, classes):
    print("Accuracy:", accuracy_score(y_true, y_preds))
    print()

    confmatrix = confusion_matrix(y_true, y_preds)
    frame = {
        f"Pred {clas}": confmatrix[:, i] for i, clas in enumerate(classes)
    }
    df_confmatrix = pd.DataFrame(frame)
    for i, clas in enumerate(classes):
        df_confmatrix.rename({i: f"Actual {clas}"}, inplace=True)
    print("Confusion matrix:")
    print(df_confmatrix)
    print()

    print("Classification report:")
    classrep = classification_report(y_true, y_preds)
    print(classrep)
    print()

    for i, clas in enumerate(classes):
        total_row = confmatrix[i].sum()
        tru = confmatrix[i][i]

        total_col = confmatrix[:, i].sum()
        fls = total_col - tru

        print(f"True {clas} rate: {tru / total_row: .3f}")
        print(f"False {clas} rate: {fls / total_col: .3f}")
        print()


def random_forest_fit_and_predict(X_train, y_train, X_test, y_test, **kwargs):
    rf = RandomForestClassifier(**kwargs)
    rf.fit(X_train, y_train)

    preds_train = rf.predict(X_train)
    preds_test = rf.predict(X_test)

    return preds_train, preds_test, rf
