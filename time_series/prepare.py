import pandas as pd


def date_to_datetime(date_col):
    return pd.to_datetime(date_col)


def localize_sale_date(df: pd.DataFrame) -> pd.DataFrame:
    date_format = "%a, %d %b %Y %H:%M:%S %Z"
    localized_sale_date = pd.to_datetime(
        df.sale_date, format=date_format, utc=True
    )
    return df.assign(sale_date=localized_sale_date)


def reassign_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("sale_date").set_index("sale_date")


def split_date_series(series: pd.Series) -> tuple:
    """
    Return six columns: year, quarter, month, day, weekday #, and whether it's a weekday (1) or weekend (0)
    """
    year = series.dt.year.rename("year")
    quarter = series.dt.quarter.rename("quarter")
    month = series.dt.month.rename("month")
    day = series.dt.day.rename("day")
    weekday = series.dt.day_name().str[:3].rename("weekday")
    is_weekday = (series.dt.weekday < 5).rename("is_weekday")
    return year, quarter, month, day, weekday, is_weekday


def add_extra_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, *split_date_series(df.sale_date)], axis=1)


def rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"sale_amount": "sale_quantity"})
    return out


def add_sales_total(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(sales_total=df.sale_quantity * df.item_price)


def sales_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["weekday"])["sales_total", "sale_quantity"].agg(
        ["sum", "median"]
    )


def change_in_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("sale_date")["sale_date", "sales_total"].sum().diff()


def prep_store_data(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(localize_sale_date)
        .pipe(add_extra_date_cols)
        .pipe(rename_cols)
        .pipe(add_sales_total)
        .pipe(reassign_index)
    )
