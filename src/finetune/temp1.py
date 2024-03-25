import json 
from ..model.roberta import RobertaClass
import torch

class Model():
    def __init__(self,metadata_path) -> None:
        self.metadata_path = metadata_path
        try:
            ''' The json file indicating all the required details should present in the metadata_path
                The incurrect path or format can raise errors.
            '''
            with open(self.metadata_path, 'r') as metadata_file:
                data = json.load(metadata_file)
        except FileNotFoundError:
            print("File not found.")
        except json.JSONDecodeError:
            print("Invalid JSON data in the file.") 

        self.state_dict_path = data['latest']['path']

    
    def load_model(self):
        checkpoint = torch.load(self.state_dict_path)
        model = RobertaClass()
        model.load_state_dict(checkpoint['model_state_dict'])
        # for param_tensor in model.state_dict():
        #     print(param_tensor)
        # model.eval()
        return model
