import torch
from transformers import RobertaTokenizer


class Model():
    ''' Actual implementation to load the model. 
    '''
    def __init__(self) -> None:
        self.model = None

    def get_model(self,saved_model_path):
        loaded_model = torch.load(saved_model_path,map_location=torch.device('cpu'))
        self.model = loaded_model
        return self.model
    
    def get_model_tokenizer(self):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
        return tokenizer
