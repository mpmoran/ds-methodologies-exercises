import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def prep_iris(df_iris):
    df = df_iris.copy()
    
    df = df.drop(columns=["species_id", "measurement_id"])
    
    df = df.rename(index=str, columns={"species_name": "species"})
    
    encoder = LabelEncoder()
    encoder.fit(df.species)
    df = df.assign(species_encode=encoder.transform(df.species))
    
    return df


def split_titanic(df_titanic):
    return train_test_split(df_titanic, train_size=0.7, random_state=123, stratify=df_titanic[["survived"]])


def min_max_scale_titanic(df_train, df_test):
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(df_train[['age', 'fare']])
    df_train_scaled[["age", "fare"]] = scaler.transform(df_train[["age", "fare"]])
    df_test_scaled[["age", "fare"]] = scaler.transform(df_test[["age", "fare"]])
    
    return df_train_scaled, df_test_scaled


def prep_titanic(df_titanic):
    df = df_titanic.copy()
    
    df.embarked = df.embarked.fillna("U")
    df.embark_town = df.embark_town.fillna("Unknown")
    
    df = df.drop(columns="deck")
    
    encoder_embark = LabelEncoder()
    encoder_embark.fit(df.embarked)
    df = df.assign(embarked_encode=encoder_embark.transform(df.embarked))
    
    encoder_sex = LabelEncoder()
    encoder_sex.fit(df.sex)
    df = df.assign(sex_encode=encoder_sex.transform(df.sex))
    
    df = df.dropna()
    
    return df
