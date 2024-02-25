import torch
from transformers import RobertaTokenizer
from roberta import RobertaClass

class Model():
    def __init__(self) -> None:
        self.model = None

    def get_model(self,saved_model_path)->RobertaClass:
        loaded_model= RobertaClass()
        loaded_model = torch.load(saved_model_path,map_location=torch.device('cpu'))
        self.model = loaded_model
        return self.model
    
    def get_model_tokenizer(self):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
        return tokenizer
