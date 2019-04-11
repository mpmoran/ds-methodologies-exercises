import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def df_plot_numeric(df: pd.DataFrame, cols: list, hue=None) -> None:
    """
    For plotting purely continuous v. purely continuous
    """
    sns.pairplot(data=df, vars=cols, hue=hue)
    plt.show()
    plt.figure(figsize=(15, 15))
    sns.heatmap(df[cols].corr(), cmap="Blues", annot=True)
    plt.show()


def df_jitter_plot(df: pd.DataFrame, X: list, Y: list, hue=None) -> None:
    """
    For plotting semi-continuous semi-categorical variables v. continuous
    """
    sns.pairplot(data=df, x_vars=X, y_vars=Y, hue=hue)
    plt.show()


def relplot_num_and_cat(
    df: pd.DataFrame, x: str, y: str, hue: str
) -> pd.DataFrame:
    """
    Write a function that will use seaborn's relplot to plot 2 numeric (ordered) variables
    and 1 categorical variable. It will take, as input, a dataframe, column name indicated
    for each of the following: x, y, & hue.
    """
    sns.relplot(x=x, y=y, hue=hue, data=df, alpha=0.8)
    plt.show
