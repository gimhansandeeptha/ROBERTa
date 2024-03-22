import json
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def load_data():
    json_file_path = "data\\Software.json"
    with open(json_file_path) as f:
        data = [json.loads(line) for line in f]

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Select only the desired columns
    df = df[['overall', 'reviewText', 'summary']]
    return df

def get_required_data(df):
    # Define the desired counts for each label
    counts = {'1.0': 64, '2.0': 64, '3.0': 128, '4.0': 64, '5.0': 64} # 5000
    new_dfs =[]

    # Sample data for each label and store in the list
    for label, count in counts.items():
        label_df = df[df['overall'] == float(label)]
        sampled_df = label_df.sample(n=count, replace=True, random_state=42)
        new_dfs.append(sampled_df)

    # Concatenate all sampled DataFrames outside the loop
    new_df = pd.concat(new_dfs, ignore_index=True)

    # Assume that 0 -> Negative, 1 -> 'Neutral' and 2 -> 'Positive'
    label_mapping = {1.0: 0, 2.0: 0, 3.0: 1, 4.0: 2, 5.0: 2}

    new_df['overall'] = new_df['overall'].replace(label_mapping)
    shuffled_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(shuffled_df.head())
    return shuffled_df

def split(df):
    X,y = df['reviewText'].values,df['overall'].values
    x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Split the temporary set into testing and validation sets
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    return x_train, y_train, x_test, y_test, x_val, y_val

class Main:
    def __init__(self) -> None:
        pass