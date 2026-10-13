import pandas as pd
import numpy as np

data = pd.read_csv("Dataset/Dataset.csv")
data = data.replace("Not Given", np.nan)

data["date_added"] = pd.to_datetime(data["date_added"], format = "%d/%m/%Y", errors = "coerce") #convert data_added column to standard datetime format
data["year_added"] = data["date_added"].dt.year #Extract only the years
data["duration_no"] = data["duration"].str.split(" ").str[0].astype(float) #Split the duration column into a number and unit
data["duration_unit"] = data["duration"].str.split(" ").str[1] # "90 min" becomes 90 and "min" becomes unit 

data = data.drop_duplicates(subset=["title", "type", "release_year"]) # Removing duplicates movies/TVshows

print("Missing Data:\n" ,data.isna().sum())
print("\nMovie vs TV show split:\n", data["type"].value_counts(normalize = True))

