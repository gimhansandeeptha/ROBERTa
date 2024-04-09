import torch
import pandas as pd
import json
import os
from datetime import datetime
from src.model.roberta import RobertaClass
from src.train.processor import get_device
from src.train.model import BuildModel
from src.train.data import data
from src.preprocess.main import DataHandler
from src.finetune.roberta_finetune import RobertaFinetune
from transformers import RobertaTokenizer
from src.database.main import Database

METADATA_PATH = "metadata/state_dict.json"
class Handler:
    def __init__(self) -> None:
        self.entry_count_per_label= 32 # Put 500 for production
        self.maximum_entry_count_per_label=self.entry_count_per_label*7
        self.metadata_path = METADATA_PATH
        self.db = Database()
        self.df: pd.DataFrame|None = None
    
    def _get_finetuned_count(self) -> int:
        with open(self.metadata_path, 'r') as file:
            data: dict = json.load(file)
        return data.get("latest",{}).get("finetune_count")
    
    def _update_finetuned_count(self, new_value):
        # Read existing JSON data from the file
        with open(self.metadata_path, 'r') as file:
            data = json.load(file)
        data["latest"]["finetune_count"] = new_value
        with open(self.metadata_path, 'w') as file:
            json.dump(data, file, indent=4)

    def _delete_excessive_data(self):
        # Delete the oldest sentiment data from gpt table if that exceeds a limit
        sentiment_count = self.db.get_sentiment_category_count()
        for sentiment, count in sentiment_count.items():
            count_to_delete = count - self.maximum_entry_count_per_label
            if count_to_delete > 0:
                self.db.delete_excessive_gpt_data(sentiment, count_to_delete)

    def _map_sentiment(self, sentiment):
        if sentiment == 'Negative':
            return 0
        elif sentiment == 'Neutral':
            return 1
        elif sentiment == 'Positive':
            return 2
        else:
            print(f"Illegal sentiment: {sentiment}")
            return None
        
    def _load_data(self): # count represent the class count (like 500 datapoints)
        '''Fine tuning condiitons: ##Update not correct now
            There need to be adiquate amount of data. (1000 data points)
            There should be class balence with 20% from the each class label
            The data points should be under(1500)

            This function is used as private function call by the functions inside the class to load the data
            This assumes the exsistance of 3 classes
        '''
        result = self.db.get_gpt_entries(self.entry_count_per_label)
        df = None
        if result is not None:
            df = pd.DataFrame(result, columns=['id','text', 'sentiment', 'datetime']) 
            df['sentiment'] = df['sentiment'].apply(self._map_sentiment)
            df.dropna(inplace=True) 
        self.df = df

    def finetune(self):
        """ This perfroms either finetuning or validating."""
        # finetuning or validating call
        # Model handling 
        # Model saving
        # Delete used data 

        self._delete_excessive_data()
        self._load_data()

        if self.df is not None:
            finetuned_count = self._get_finetuned_count()
            if finetuned_count < 5:
                finetune = FineTune()
                finetune.finetune(self.df)
                self._update_finetuned_count(finetuned_count+1)
                finetune.save_checkpoint(self.metadata_path)
            else:
                validate = Validate()
                validate.model_validate('insert actual checkpoint path', self.df)
                validate.save_checkpoint(self.metadata_path)


class FineTune:
    def __init__(self) -> None:
        self.tuned_model = None 
        self.finetuned_obj = None

    def finetune(self, df):
        if df is not None:
            ## Change the hyperparameter retreval
            data_handler = DataHandler(df,{"train_size":1.0, "test_size":0, "validation_size": 0.0})
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, do_lower_case=True)
            train, test, val = data_handler.get_dataloaders(tokenizer,256,32,16)
            model = RobertaClass(hidden_size=768,dropout_prob=0.3, num_classes=3)
            loss_function = torch.nn.CrossEntropyLoss()
            LEARNING_RATE = 1e-05
            optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
            finetune = RobertaFinetune(model=model,optimizer=optimizer,loss_function=loss_function)
            self.tuned_model = finetune.finetune(training_loader=train,validation_loader=val)
            self.finetuned_obj = finetune

            # Code to remove the data points used in the database goes here. 
        else:
            pass
        # return get_required_data(load_data())  # For testing
    
    def save_checkpoint(self, metadata_path):
        with open(metadata_path, 'r') as file:
            data = json.load(file)
        finetune_count = data.get("latest", {}).get("finetune_count")

        checkpoint_path = data.get("latest", {}).get("path")
        folder_path = os.path.dirname(checkpoint_path)
        # file_name = os.path.basename(checkpoint_path)
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_checkpoint_path = os.path.join(folder_path, f"{current_datetime}.pth")
        self.finetuned_obj.save_checkpoint(new_checkpoint_path)
        
        # Saving the new metadata about the checkpoint.
        data["latest"]["path"] = new_checkpoint_path
        data["latest"]["finetune_count"] = finetune_count+1
        with open(metadata_path, 'w') as file:
            json.dump(data, file, indent=4)

        # Removeing the exsisting file
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"{checkpoint_path} has been successfully deleted.")
        else:
            print(f"The file {checkpoint_path} does not exist.")


class Validate:
    def __init__(self) -> None:
        self.file_path = "models\\stable\\stable_checkpoint.pth"
        self.metadata_path = "metadata\\roberta.json"
        self.finetuned_obj: RobertaFinetune|None = None
        self.validation_loss = float("inf")
        self.validated_model = None

    def model_validate(self, checkpoint_path, df):
        # Load the model
        self.validated_model=RobertaClass(hidden_size=768,dropout_prob=0.3, num_classes=3)
        checkpoint=torch.load(checkpoint_path)
        self.validated_model.load_state_dict(checkpoint.get("model_state_dict"))

        # Load the optimizer
        LEARNING_RATE = 1e-05        
        optimizer =torch.optim.Adam(params=self.validated_model.parameters(), lr=LEARNING_RATE)
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict"))

        # Preprocess the data (To be able for training)
        data_handler = DataHandler(df,{"train_size":0.0, "test_size":0.0, "validation_size": 1.0})
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, do_lower_case=True)
        _, _, val = data_handler.get_dataloaders(tokenizer,256,32,16)

        # validate
        loss_function = torch.nn.CrossEntropyLoss()
        self.finetuned_obj = RobertaFinetune(model=self.validated_model,optimizer=optimizer,loss_function=loss_function)
        self.validation_loss=self.finetuned_obj.validate(val, "cpu")

    def save_checkpoint(self):
        with open(self.metadata_path, 'r') as file:
            data = json.load(file)
        saved_model_validation_loss = float(data.get("model", {}).get("validation_loss"))
        if self.validation_loss <= saved_model_validation_loss: 
            checkpoint_path = data.get("model", {}).get("trained_model_path")
            folder_path = os.path.dirname(checkpoint_path)
            # file_name = os.path.basename(checkpoint_path)
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            new_checkpoint_path = os.path.join(folder_path, f"{current_datetime}.pth")
            self.finetuned_obj.save_checkpoint(new_checkpoint_path)
        
            data["model"]["trained_model_path"] = new_checkpoint_path
            data["model"]["validation_loss"] = self.validation_loss
            with open(self.metadata_path, "w") as file:
                json.dump(data, file, indent=4)
            
            # Removeing the exsisting file
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"{checkpoint_path} has been successfully deleted.")
            else:
                print(f"The file {checkpoint_path} does not exist.")



### --------------------Testing--------------------- 
# def load_data():
#     import json
#     import pandas as pd

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
#     import pandas as pd
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

# def run_finetune(df):
#     # df = get_required_data(load_data())
#     data_handler = DataHandler(df,{"train_size":0.7, "test_size":0, "validation_size": 0.3})
#     tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, do_lower_case=True)
#     train, test, val = data_handler.get_dataloaders(tokenizer,256,32,16)
#     print(train, test, val)

#     LEARNING_RATE = 1e-05
#     MAX_LEN = 256
#     TRAIN_BATCH_SIZE = 32
#     VALID_BATCH_SIZE = 16
#     model = RobertaClass(hidden_size=768,dropout_prob=0.3, num_classes=3)
#     loss_function = torch.nn.CrossEntropyLoss()
#     optimizer =torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
#     finetune = RobertaFinetune(model=model,optimizer=optimizer,loss_function=loss_function)
#     finetune.finetune(training_loader=train,validation_loader=val)



# tune = FineTune()
# tune.finetune()
# validate = Validate()
    

# # Unit testing Load data
# fine_tune = FineTune()
# result = fine_tune._load_data(10)
# print(result.head())

