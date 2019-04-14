from typing import Dict, List, Tuple
from collections import namedtuple
import io
from copy import deepcopy

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
    mean_absolute_error,
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

# Write a function that accepts a series (i.e. one column from a data frame) and summarizes how many outliers are in the series. This function should accept a second parameter that determines how outliers are detected, with the ability to detect outliers in 3 ways:

# Using the IQR
# Using standard deviations
# Based on whether the observation is in the top or bottom 1%.
# Use your function defined above to identify columns where you should handle the outliers.

# Write a function that accepts the zillow data frame and removes the outliers. You should make a decision and document how you will remove outliers.

# Is there erroneous data you have found that you need to remove or repair? If so, take action.

# Are there outliers you want to "squeeze in" to a max value? (e.g. all bathrooms > 6 => bathrooms = 6). If so, make those changes.


class DataSet(object):
    def __init__(self):
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.xcols = None
        self.ycol = None
        self.model = None
        self.pred_train = None
        self.pred_test = None
        self.score = None
        self.confmatrix = None
        self.classrep = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.support = None

    def get_train(self):
        return pd.concat([self.X_train, self.y_train], axis=1)

    train = property(get_train)

    def get_test(self):
        return pd.concat([self.X_test, self.y_test], axis=1)

    test = property(get_test)

    def data_only_copy(self):
        """
        returns a new object with only the data copied
        """
        copy = DataSet()
        copy.df = self.df.copy()

        return copy


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


# LINEAR REGRESSION


LinRegModel = namedtuple(
    "LinRegModel", ["model", "pred_train", "pred_test", "equation"]
)
# model is a LinearRegression object
# pred_train is a numpy array
# pred_test is a numpy array
# equation is a tuple of str, float, float, str


def linreg_print_model(
    y_label: str, y_intercept: float, coefs: List[float], X_labels: List[str]
) -> None:
    """
    print equation of a linear regression model
    """
    # print(coefs)
    # print(X_labels)
    print(f"{y_label} = {y_intercept:.2f}", end="")
    for coef, label in zip(coefs, X_labels):
        print(f" +\n\t{coef:.3f} * {label}", end="")
    print()


def linreg_fit_and_predict(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs,
) -> namedtuple:
    """
    perform linear regression on train data and predict on the training
    and test data

    return predictions of the train and test data
    """
    lm = LinearRegression(**kwargs)
    lm.fit(X_train, y_train)

    pred_train = lm.predict(X_train)
    pred_test = lm.predict(X_test)

    y_label = y_train.columns[0]
    if isinstance(lm.intercept_, float):
        y_intercept = lm.intercept_
    else:
        y_intercept = lm.intercept_[0]
    coefs = lm.coef_[0]
    X_labels = X_train.columns

    return LinRegModel(
        lm, pred_train, pred_test, (y_label, y_intercept, coefs, X_labels)
    )


def linreg_evaluate_model(
    X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray
) -> None:
    """
    print evaluation metrics for linear regression model
    """
    y_label = y.columns[0]
    X_labels = X.columns

    print("Univariate Linear Regression Model Evaluation")
    meanse = mean_squared_error(y, y_pred)
    print(f"\tMean SE: {meanse:.3f}")

    meanae = mean_absolute_error(y, y_pred)
    print(f"\tMean AE: {meanae:.3f}")

    print()

    medianae = median_absolute_error(y, y_pred)
    print(f"\tMedian AE: {medianae:.3f}")

    print()

    r2 = r2_score(y, y_pred)
    print(
        f"\t{r2:.2%} of the variance in {y_label} can be explained "
        f"by {X_labels.tolist()}."
    )
    print()

    print("P-VALUE")
    f_vals, p_vals = f_regression(X, y)
    print(f"\tTrain: {p_vals[0]:.3}")


def linreg_plot_residuals(y_test: pd.Series, pred_test: np.ndarray):
    """
    plot residuals for a linear regression model
    """
    y_label = y_test.columns[0]
    plt.scatter(pred_test, pred_test - y_test, c="g", s=20)
    plt.hlines(y=0, xmin=pred_test.min(), xmax=pred_test.max())
    plt.title("Residual plot")
    plt.ylabel("Residuals")
    plt.xlabel(y_label)
    plt.show()


def linreg_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    **kwargs,
):
    """
    models, evaluates (train and test), and plots residuals (test only)
    for linear regression model
    """
    lrm = linreg_fit_and_predict(X_train, X_test, y_train, y_test, **kwargs)

    # TODO: print equation using linreg_print_model()
    print("EQUATION")
    linreg_print_model(
        lrm.equation[0], lrm.equation[1], lrm.equation[2], lrm.equation[3]
    )

    print("-" * 50)

    print("TRAIN")
    linreg_evaluate_model(X_train, y_train, lrm.pred_train)

    print("-" * 50)

    print("TEST")
    linreg_evaluate_model(X_test, y_test, lrm.pred_test)

    linreg_plot_residuals(y_test, lrm.pred_test)


# def multi_linreg_evaluate_model(
#     X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray
# ) -> None:
#     """
#     print evaluation metrics for linear regression model
#     """
#     y_label = y.columns[0]
#     X_labels = X.columns[0]

#     print("Multivariate Linear Regression Model Evaluation")
#     meanse = mean_squared_error(y, y_pred)
#     print(f"\tMean SE: {meanse:.3f}")

#     meanae = mean_absolute_error(y, y_pred)
#     print(f"\tMean AE: {meanae:.3f}")

#     print()

#     medianae = median_absolute_error(y, y_pred)
#     print(f"\tMedian AE: {medianae:.3f}")

#     print()

#     r2 = r2_score(y, y_pred)
#     print(
#         f"\t{r2:.2%} of the variance in {y_label} can be explained "
#         f"by {X_labels.tolist()}."
#     )
#     print()

#     print("P-VALUE")
#     f_vals, p_vals = f_regression(X, y)
#     print(f"\tTrain: {p_vals[0]:.3}")
#     print()


# def multi_linreg_fit_and_predict(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     y_train: pd.Series,
#     y_test: pd.Series,
#     **kwargs,
# ) -> namedtuple:
#     """
#     perform multivariate linear regression on train data and predict on the
#     training and test data

#     return predictions of the train and test data
#     """
#     lm = LinearRegression(**kwargs)
#     lm.fit(X_train, y_train)

#     pred_train = lm.predict(X_train)
#     pred_test = lm.predict(X_test)

#     y_label = y_train.columns[0]
#     y_intercept = lm.intercept_[0]
#     coefs = lm.coef_[0]
#     X_labels = X_train.columns

#     return LinRegModel(
#         lm, pred_train, pred_test, (y_label, y_intercept, coefs, X_labels)
#     )


# def multi_linreg_model(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     y_train: pd.Series,
#     y_test: pd.Series,
#     **kwargs,
# ):
#     """
#     models, evaluates (train and test), and plots residuals (test only)
#     for a multi linear regression model
#     """
#     lrm = multi_linreg_fit_and_predict(X_train, X_test, y_train, y_test, **kwargs)

#     linreg_print_model(lrm.equation[0], lrm.equation[1], lrm.equation[])

#     print("TRAIN")
#     multi_linreg_evaluate_model(X_train, y_train, lrm.pred_train)

#     print("-" * 50)

#     print("TEST")
#     multi_linreg_evaluate_model(X_test, y_test, lrm.pred_test)

#     linreg_plot_residuals(y_test, lrm.pred_test)


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
