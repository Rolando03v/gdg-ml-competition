import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load dataset and return cleaned and preprocessed features and target
# TODO: use more of the columns, preprocess them in a different way,
#       or even include new features (e.g. from other datasets)

# columns: 'Poster_Link', 'Series_Title', 'Released_Year', 'Certificate',
#          'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director',
#          'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes'

# columns used in template: 'Released_Year', 'Certificate', 'Runtime', 'Genre' (kind of',
# 'IMDB_Rating', 'Meta_score', 'No_of_Votes'
def get_prepared_data(data_path="data"):

    # Load raw data
    # this function tries to combine all .csv files in the data folder
    # it matches them up using the "Series_Title" column
    # if you want to use additional datasets, make sure they have a "Series_Title" column
    # if not, you will need additional logic to join the datasets
    # do not rename the column by hand, add code before this point to rename it
    # remember: we will not manually modify your datasets, so your code must do any formatting automatically
    data = get_raw_data(data_path)

    if data.empty:
        print("No valid data found.")
        return None, None
    
    if "Gross" not in data.columns:
        print("Error: 'Gross' column missing in dataset.")
        return None, None

    # Drop columns in text format (not used in the demo, may be useful to you)
    drop_cols = ["Poster_Link", "Series_Title", "Overview", "Director", "Star1", "Star2", "Star3", "Star4"]
    data.drop(columns=drop_cols, inplace=True, errors='ignore')

    data.ffill(inplace = True)
    # take only the first genre from the list of genres (you might want to do something more sophisticated)
    data["Genre"] = data["Genre"].apply(lambda x: x.split(",")[0] if isinstance(x, str) else "Unknown")

    # convert "Gross" into a number (remove ",")
    data["Gross"] = data["Gross"].replace({',': ''}, regex=True).apply(pd.to_numeric, errors='coerce')

   # One-hot encode categorical columns
    categorical_features = ["Genre", "Certificate"]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorical_encoded = encoder.fit_transform(data[categorical_features])
    categorical_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(), index=data.index)
    data.drop(columns=categorical_features, inplace=True)
    data = pd.concat([data, categorical_df], axis=1)


    # Normalize numerical features
    numeric_features = ["Released_Year", "Runtime", "IMDB_Rating", "Meta_score", "No_of_Votes"]

    data[numeric_features] = data[numeric_features].apply(pd.to_numeric, errors='coerce')

    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())

    scaler = StandardScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    # Define features and target
    features = data.drop(columns=["Gross"], errors='ignore')
    target = data["Gross"] if "Gross" in data.columns else None

    # Convert to PyTorch tensors
    features = torch.tensor(features.values, dtype=torch.float32)
    target = torch.tensor(target.values, dtype=torch.float32) if target is not None else None
    
    return features, target

def get_all_titles(data_path="data"):
    data = get_raw_data(data_path)
    return data["Series_Title"] if "Series_Title" in data.columns else pd.Series([])

def get_raw_data(path="data"):
    # read in every csv file, join on "Series_Title"
    # return the raw data
    if not os.path.exists(path):
        print(f"Data directory '{path}' not found.")
        return pd.DataFrame()  # return empty DataFrame
    
    files = os.listdir(path)
    if not files:
        print(f"No CSV files found in the directory '{path}'.")
        return pd.DataFrame()  # return empty DataFrame
    data = pd.DataFrame()
    for file in files:
        if file.endswith(".csv"):
            print(f"Loading file: {file}")
            df = pd.read_csv(os.path.join(path, file))
            
            # Check if the 'Series_Title' column exists
            if "Series_Title" not in df.columns:
                print(f"Warning: 'Series_Title' column missing in {file}. Skipping this file.")
                continue
            
            df.dropna(subset=["Series_Title"], inplace=True)
            # join on "Series_Title"
            if data.empty:
                data = df
            else:
                data = data.merge(df, on="Series_Title", how="outer")
    if data.empty:
        print("No data found after merging CSV files.")
    else:
        print(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns.")

    return data
if __name__ == "__main__":
    features, target = get_prepared_data()

    if features is not None and target is not None:
        print(f"Features shape: {features.shape}")
        print(f"Target shape: {target.shape}")
    else:
        print("Failed to load data.")
