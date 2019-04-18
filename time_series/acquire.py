import os
import requests
import pandas as pd


def get_from_zach_api(table) -> pd.DataFrame:
    root_endpoint = "https://python.zach.lol"
    path = f"/api/v1/{table}"

    data = requests.get(root_endpoint + path).json()
    max_page = data["payload"]["max_page"]
    entries = []
    for i in range(1, max_page + 1):
        url = root_endpoint + path + f"?page={i}"
        data = requests.get(url).json()
        entries += data["payload"][table]

    return pd.DataFrame(entries)


def get_table(table_name):
    csv_fname = f"{table_name}.csv"
    if os.path.exists(csv_fname):
        print(f"Reading {csv_fname} . . . .")
        return pd.read_csv(csv_fname)
    else:
        df_table = get_from_zach_api(table_name)
        df_table.to_csv(csv_fname, index=False)
        return df_table


def get_items() -> pd.DataFrame:
    return get_table("items")


def get_stores() -> pd.DataFrame:
    return get_table("stores")


def get_sales() -> pd.DataFrame:
    return get_table("sales")


def get_product_info() -> pd.DataFrame:
    items = get_items()
    stores = get_stores()
    sales = get_sales()

    merged = sales.merge(
        items, how="left", left_on="item", right_on="item_id"
    ).merge(stores, how="left", left_on="store", right_on="store_id")

    return merged.drop(columns=["item", "store"])
