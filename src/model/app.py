from ..inference.inference import Inference
from .model import Model
import json

class App():
    ''' Model initilization and inferncing. Generic class for all Roberta models.
    '''
    def __init__(self, metadata_path):  
        try:
            ''' The json file indicating all the required details should present in the metadata_path
                The incurrect path or format can raise errors.
            '''
            with open(metadata_path, 'r') as metadata_file:
                data = json.load(metadata_file)
        except FileNotFoundError:
            print("File not found.")
        except json.JSONDecodeError:
            print("Invalid JSON data in the file.") 

        self.model_file_path = data['model']['trained_model_path']
        self.max_len = data['model']['max_len'] 
        self.tokenizer_name = data['model']['tokenizer']
        self.device = 'cpu'                                        ### change 
        self.model = None 
        self.inference = None

    def start_model(self):
        '''' Load the saved model 
             After this is done model is redy for inferencing.
        '''
        self.model = Model()
        saved_model = self.model.get_model(self.model_file_path)
        tokenizer = self.model.get_model_tokenizer(self.tokenizer_name)
        self.inference = Inference(saved_model, tokenizer, self.max_len, self.device)

    def predict(self, text):
        return self.inference.inference(text)
