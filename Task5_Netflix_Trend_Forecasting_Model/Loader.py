import pandas as pd
import numpy as np


def load_data(filepath):
    # using the file path from the source
    data = pd.read_csv(filepath, lineterminator='\n')
    data = data.replace("Not Given", np.nan)

    # Strip hidden characters from column names
    data.columns = data.columns.str.strip()

    # converting data_added column to standard datetime format
    data["date_added"] = pd.to_datetime(
        data["date_added"].str.strip(), format="%d/%m/%Y", errors="coerce")

    data["year_added"] = data["date_added"].dt.year  # Extract only the years

    data["duration_no"] = data["duration"].str.split(" ").str[0].astype(
        float)  # Split the duration column into a number and unit

    data["duration_unit"] = data["duration"].str.split(
        " ").str[1]  # "90 min" becomes 90 and "min" becomes unit

    # Removing duplicates movies/TVshows
    data = data.dropna(subset=['date_added'])
    data = data.drop_duplicates(subset=["title", "type", "release_year"])

    print("Missing Data:\n", data.isna().sum())
    print("\nMovie vs TV show split:\n",
          data["type"].value_counts(normalize=True))

    return data
