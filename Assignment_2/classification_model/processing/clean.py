import re

import numpy as np
import pandas as pd


def _get_first_cabin(row):
    try:
        return row.split()[0]
    except Exception:
        return np.nan


def _get_title(passenger) -> str:
    line = passenger
    if re.search("Mrs", line):
        return "Mrs"
    elif re.search("Mr", line):
        return "Mr"
    elif re.search("Miss", line):
        return "Miss"
    elif re.search("Master", line):
        return "Master"
    else:
        return "Other"


def clean_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    df = df.replace("?", np.nan)
    df["cabin"] = df["cabin"].apply(_get_first_cabin)
    df["title"] = df["name"].apply(_get_title)
    df["fare"] = df["fare"].astype("float")
    df["age"] = df["age"].astype("float")
    df.drop(
        labels=["name", "ticket", "boat", "body", "home.dest"], axis=1, inplace=True
    )
    return df
