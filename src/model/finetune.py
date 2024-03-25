import torch
from .roberta import RobertaClass

class FineTune():
    def __init__(self, model_dict_path) -> None:
        self.model_dict_path = model_dict_path

    def load_model(self):
        checkpoint = torch.load(self.model_dict_path)
        model = RobertaClass()
        model.load_state_dict(checkpoint['model_state_dict'])
        for param_tensor in model.state_dict():
            print(param_tensor)
        # model.eval()
        return model
