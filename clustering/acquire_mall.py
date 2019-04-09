from env import host, user, password
import pandas as pd


def get_db_url(
    hostname: str, username: str, password: str, db_name: str
) -> str:
    """
    return url for accessing a mysql database
    """
    return f"mysql+pymysql://{username}:{password}@{hostname}/{db_name}"


def df_from_sql(query: str, url: str) -> pd.DataFrame:
    """
    return a Pandas DataFrame resulting from a sql query
    """
    return pd.read_sql(query, url)


def get_mallcustomer_data() -> pd.DataFrame:
    db = "mall_customers"
    query = "SELECT * FROM customers;"

    url = get_db_url(host, user, password, db)
    return df_from_sql(query, url)
