from .roberta_Sentiment_data import RobertaSentimentData
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from transformers import RobertaTokenizer

class DataHandler():
    def __init__(self,df,split_dict:dict=None) -> None:
        columns_list = df.columns.tolist()
        # Check if both "text" and "sentiment" are present in the column names
        if "text" not in columns_list or "sentiment" not in columns_list:
            raise ValueError("DataFrame is missing 'text' column or 'sentiment' column ")
        self.df = df

        self.train_size = 0.7
        self.test_size = 0.15
        self.validation_size = 0.15

        if split_dict is not None:
            self.train_size = split_dict.get("train_size",0)
            self.test_size = split_dict.get("test_size",0)
            self.validation_size = split_dict.get("validation_size",0)

        split_sum  = self.train_size+self.test_size+self.validation_size
        if split_sum != 1:
            raise ValueError(f"Split sizes should sum up to 1 but sum is {split_sum}")

    def split(self, df):
        X,y = df['text'].values,df['sentiment'].values
        x_train, x_temp, y_train, y_temp = train_test_split(X, y, train_size=self.train_size, stratify=y, random_state=42)

        # Split the temporary set into testing and validation sets
        test_size = round((self.test_size/(self.test_size+self.validation_size)),1)
        if test_size == 0:
            x_val, y_val  = x_temp, y_temp
            x_test, y_test = None, None
        elif test_size == 1:
            x_val, y_val  = None, None 
            x_test, y_test = x_temp, y_temp

        else:
            x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, train_size=test_size, stratify=y_temp, random_state=42)

        return x_train, y_train, x_test, y_test, x_val, y_val
    
    def get_dataloaders(self, tokenizer, max_len, train_batch_size=None, validation_batch_size=None):
        x_train, y_train, x_test, y_test, x_val, y_val = self.split(self.df)
        training_loader = testing_loader = validation_loader = None

        if x_train is not None:
            training_set = RobertaSentimentData(x_train, y_train, tokenizer, max_len)
            train_params = {'batch_size': train_batch_size,
                        'shuffle': True,
                        'num_workers': 0
                        }
            training_loader = DataLoader(training_set, **train_params)

        if x_test is not None:
            testing_set = RobertaSentimentData(x_test, y_test, tokenizer, max_len)
            test_params = {'batch_size': 1,
                        'shuffle': False,
                        'num_workers': 0
                        }
            testing_loader = DataLoader(testing_set, **test_params)

        if x_val is not None:
            validation_set = RobertaSentimentData(x_val, y_val, tokenizer, max_len)
            validation_params = {'batch_size': validation_batch_size,
                        'shuffle': True,
                        'num_workers': 0
                        }
            validation_loader = DataLoader(validation_set, **validation_params)

        

        return training_loader, testing_loader, validation_loader


# def load_data():
#     json_file_path = "data\\Software.json"
#     with open(json_file_path) as f:
#         data = [json.loads(line) for line in f]

#     # Create a DataFrame
#     df = pd.DataFrame(data)

#     # Select only the desired columns
#     df = df[['overall', 'reviewText', 'summary']]
#     df.rename(columns={'overall': 'sentiment', 'reviewText': 'text'}, inplace=True)
#     return df

# def get_required_data(df):
#     # Define the desired counts for each label
#     counts = {'1.0': 64, '2.0': 64, '3.0': 128, '4.0': 64, '5.0': 64} # 5000
#     new_dfs =[]

#     # Sample data for each label and store in the list
#     for label, count in counts.items():
#         label_df = df[df['sentiment'] == float(label)]
#         sampled_df = label_df.sample(n=count, replace=True, random_state=42)
#         new_dfs.append(sampled_df)

#     # Concatenate all sampled DataFrames outside the loop
#     new_df = pd.concat(new_dfs, ignore_index=True)

#     # Assume that 0 -> Negative, 1 -> 'Neutral' and 2 -> 'Positive'
#     label_mapping = {1.0: 0, 2.0: 0, 3.0: 1, 4.0: 2, 5.0: 2}

#     new_df['sentiment'] = new_df['sentiment'].replace(label_mapping)
#     shuffled_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)
#     print(shuffled_df.head())
#     return shuffled_df

# df = get_required_data(load_data())
# data_handler = DataHandler(df,{"train_size":0.7, "test_size":0.3, "validation_size": 0.0})
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, do_lower_case=True)
# train, test, val = data_handler.get_dataloaders(tokenizer,256,32,16)
# print(train, test, val)