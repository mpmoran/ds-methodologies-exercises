from typing import Tuple
from env import host, user, password
import pandas as pd
from sqlalchemy import create_engine


def get_db_url(
    hostname: str, username: str, password: str, db_name: str
) -> str:
    """
    return url for accessing a mysql database
    """
    return f"mysql+pymysql://{username}:{password}@{hostname}/{db_name}"


def get_sql_conn(hostname: str, username: str, password: str, db_name: str):
    """
    return a mysql connection object
    """
    return create_engine(get_db_url(host, user, password, db_name))


def df_from_sql(query: str, url: str) -> pd.DataFrame:
    """
    return a Pandas DataFrame resulting from a sql query
    """
    return pd.read_sql(query, url)


def get_zillow_data(year: int) -> pd.DataFrame:
    db = "zillow"
    join_query = (
        "CREATE TEMPORARY TABLE ada_677.zillow_draft "
        f"SELECT properties_{year}.*, predictions_{year}.logerror, "
        f"predictions_{year}.transactiondate, "
        "airconditioningtype.airconditioningdesc, "
        "architecturalstyletype.architecturalstyledesc, "
        "buildingclasstype.buildingclassdesc, "
        "heatingorsystemtype.heatingorsystemdesc, "
        "propertylandusetype.propertylandusedesc, "
        "storytype.storydesc, typeconstructiontype.typeconstructiondesc "
        f"FROM properties_{year} "
        f"JOIN predictions_{year} USING(parcelid) "
        "LEFT JOIN airconditioningtype using(airconditioningtypeid) "
        "LEFT JOIN architecturalstyletype using(architecturalstyletypeid) "
        "LEFT JOIN buildingclasstype using(buildingclasstypeid) "
        "LEFT JOIN heatingorsystemtype using(heatingorsystemtypeid) "
        "LEFT JOIN propertylandusetype using(propertylandusetypeid) "
        "LEFT JOIN storytype using(storytypeid) "
        "LEFT JOIN typeconstructiontype using(typeconstructiontypeid); "
    )

    drop_query = (
        "ALTER TABLE ada_677.zillow_draft DROP id, "
        "DROP airconditioningtypeid, DROP architecturalstyletypeid, "
        "DROP buildingclasstypeid, DROP heatingorsystemtypeid, "
        "DROP propertylandusetypeid, DROP storytypeid, "
        "DROP typeconstructiontypeid; "
    )

    retrieve_query = "SELECT * FROM ada_677.zillow_draft;"

    engine = get_sql_conn(host, user, password, db)
    with engine.connect() as conn:
        conn.execute(join_query)
        conn.execute(drop_query)
        df = pd.read_sql(retrieve_query, conn)

    return df


def get_2016_zillow_data() -> pd.DataFrame:
    return get_zillow_data(2016)


def get_2017_zillow_data() -> pd.DataFrame:
    return get_zillow_data(2017)


def df_to_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path)


def combine_years(
    df_2016: pd.DataFrame, df_2017: pd.DataFrame
) -> pd.DataFrame:
    return df_2016.append(df_2017)


def get_zillow_from_csv(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(path)
