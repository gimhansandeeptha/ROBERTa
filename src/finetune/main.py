from ..model.roberta import RobertaClass
from ..train.processor import get_device
from ..train.model import BuildModel
from ..train.data import data
import torch

from ..preprocess.main import DataHandler
from .roberta_finetune import RobertaFinetune
from transformers import RobertaTokenizer

# device = get_device()
# model = RobertaClass(hidden_size=768,dropout_prob=0.3, num_classes=3)

# LEARNING_RATE = 1e-05
# MAX_LEN = 256
# TRAIN_BATCH_SIZE = 32
# VALID_BATCH_SIZE = 16

# df = data.load_data()
# new_df = data.get_required_data(df)

# model_build = BuildModel(model)

# loss_function = torch.nn.CrossEntropyLoss()
# optimizer =torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
# model_build.train(new_df,MAX_LEN, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, optimizer, loss_function)


def load_data():
    import json
    import pandas as pd

    json_file_path = "data\\Software.json"
    with open(json_file_path) as f:
        data = [json.loads(line) for line in f]

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Select only the desired columns
    df = df[['overall', 'reviewText', 'summary']]
    df.rename(columns={'overall': 'sentiment', 'reviewText': 'text'}, inplace=True)
    return df

def get_required_data(df):
    import pandas as pd
    # Define the desired counts for each label
    counts = {'1.0': 64, '2.0': 64, '3.0': 128, '4.0': 64, '5.0': 64} # 5000
    new_dfs =[]

    # Sample data for each label and store in the list
    for label, count in counts.items():
        label_df = df[df['sentiment'] == float(label)]
        sampled_df = label_df.sample(n=count, replace=True, random_state=42)
        new_dfs.append(sampled_df)

    # Concatenate all sampled DataFrames outside the loop
    new_df = pd.concat(new_dfs, ignore_index=True)

    # Assume that 0 -> Negative, 1 -> 'Neutral' and 2 -> 'Positive'
    label_mapping = {1.0: 0, 2.0: 0, 3.0: 1, 4.0: 2, 5.0: 2}

    new_df['sentiment'] = new_df['sentiment'].replace(label_mapping)
    shuffled_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(shuffled_df.head())
    return shuffled_df

def run_finetune():
    df = get_required_data(load_data())
    data_handler = DataHandler(df,{"train_size":0.7, "test_size":0, "validation_size": 0.3})
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, do_lower_case=True)
    train, test, val = data_handler.get_dataloaders(tokenizer,256,32,16)
    print(train, test, val)

    LEARNING_RATE = 1e-05
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    model = RobertaClass(hidden_size=768,dropout_prob=0.3, num_classes=3)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer =torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    finetune = RobertaFinetune(model=model,optimizer=optimizer,loss_function=loss_function)
    finetune.finetune(training_loader=train,validation_loader=val)
