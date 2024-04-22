import torch
from transformers import RobertaTokenizer
from src.model.roberta import RobertaClass

class Model():
    ''' Actual implementation to load the model. 
    '''
    def __init__(self) -> None:
        self.model: torch.nn.Module = None

    def get_model(self,saved_model_path):
        print(saved_model_path)
        ''' Returns the model in the indicated path.
        '''
        model = RobertaClass() # This model can be changed to different model architecture. in roberta_models folder.
        self.model = model
        
        loaded_model = torch.load(saved_model_path)   ### change 
        self.model.load_state_dict(loaded_model.get("model_state_dict"))
        return self.model
    
    def get_model_tokenizer(self, tokenizer_name:str):
        ''' extend this method to load other tokenizers. 
            tokenizer name should match exactly to the tokenizer name in the metadata file.
        '''
        tokenizer = None
        if tokenizer_name == "RobertaTokenizer_1":
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
        return tokenizer
