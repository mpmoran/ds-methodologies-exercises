import requests
import pandas as pd


def get_from_zach_api(table) -> pd.DataFrame:
    base_url = "https://python.zach.lol"
    endpoint = f"/api/v1/{table}"

    data = requests.get(base_url + endpoint).json()
    max_page = data["payload"]["max_page"]
    entries = []
    for i in range(1, max_page + 1):
        url = base_url + endpoint + f"?page={i}"
        data = requests.get(url).json()
        entries += data["payload"][table]

    return pd.DataFrame(entries)


def get_items() -> pd.DataFrame:
    return get_from_zach_api("items")


def get_stores() -> pd.DataFrame:
    return get_from_zach_api("stores")


def get_sales() -> pd.DataFrame:
    return get_from_zach_api("sales")


def download_product_info() -> None:
    get_items().to_csv("items.csv")
    get_stores().to_csv("stores.csv")
    get_sales().to_csv("sales.csv")


def read_product_info() -> tuple:
    items = pd.read_csv("items.csv", index_col=0)
    stores = pd.read_csv("stores.csv", index_col=0)
    sales = pd.read_csv("sales.csv", index_col=0)
    return items, stores, sales


def get_product_info() -> pd.DataFrame:
    items, stores, sales = read_product_info()
    return sales.merge(
        items, how="left", left_on="item", right_on="item_id"
    ).merge(stores, how="left", left_on="store", right_on="store_id")
