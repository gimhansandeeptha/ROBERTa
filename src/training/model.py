import torch
from torch.utils.data import DataLoader
from data import load_data, get_required_data, split
from transformers import RobertaModel, RobertaTokenizer
from roberta import RobertaClass
import processor
from train import robertaTrain

class RobertaSentimentData():
    '''
    Custom class for handling sentiment data. To be able to pass to the DataLoader.
    __getitem__ function do the tokenization for each text sample.
    '''
    def __init__(self, x, y, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = x
        self.targets = y
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class DataHandler():
    def __init__(self,df) -> None:
        self.df = df

    def get_dataloaders(self, tokenizer, max_len, train_batch_size, validation_batch_size):
        x_train, y_train, x_test, y_test, x_val, y_val = split(self.df)

        training_set = RobertaSentimentData(x_train, y_train, tokenizer, max_len)
        testing_set = RobertaSentimentData(x_test, y_test, tokenizer, max_len)
        validation_set = RobertaSentimentData(x_val, y_val, tokenizer, max_len)

        train_params = {'batch_size': train_batch_size,
                        'shuffle': True,
                        'num_workers': 0
                        }

        test_params = {'batch_size': 1,
                        'shuffle': False,
                        'num_workers': 0
                        }

        validation_params = {'batch_size': validation_batch_size,
                        'shuffle': True,
                        'num_workers': 0
                        }

        training_loader = DataLoader(training_set, **train_params)
        testing_loader = DataLoader(testing_set, **test_params)
        validation_loader = DataLoader(validation_set, **validation_params)

        return training_loader, testing_loader, validation_loader


class BuildModel():
    def __init__(self, model, tokenizer_name = 'roberta-base', device='cpu'):
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name, truncation=True, do_lower_case=True)
        self.device = device
        self.model = model 

    def train(self,df, learning_rate, max_len, train_batch_size,validation_batch_size):
        self.model.to(self.device)
        data_handler = DataHandler(df)
        training_loader, testing_loader, validation_loader = data_handler.get_dataloaders(self.tokenizer,max_len, train_batch_size, validation_batch_size)
        train = robertaTrain(self.model,learning_rate)
        train.train(training_loader, validation_loader,self.device)