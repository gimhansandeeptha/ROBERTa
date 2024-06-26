import torch
from transformers import RobertaTokenizer
from src.model.roberta import RobertaClass

class Model():
    ''' Actual implementation to load the model. 
    '''
    def __init__(self) -> None:
        self.model = None

    def get_model(self,saved_model_path):
        ''' Returns the model in the indicated path.
        '''
        print("src.model.model.Model")
        # loaded_model = torch.load(saved_model_path,map_location=torch.device('cpu'))   ### change 
        # self.model = loaded_model
        # return self.model

        checkpoint: dict=torch.load(saved_model_path)
        model_state_dict = checkpoint.get("model_state_dict")
        model = RobertaClass() # This model can be changed to different model architecture. in roberta_models folder.
        self.model = model
        self.model.load_state_dict(model_state_dict)
        return self.model
    
    def get_model_tokenizer(self, tokenizer_name:str):
        ''' extend this method to load other tokenizers. 
            tokenizer name should match exactly to the tokenizer name in the metadata file.
        '''
        tokenizer = None
        if tokenizer_name == "RobertaTokenizer_1":
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
        return tokenizer
