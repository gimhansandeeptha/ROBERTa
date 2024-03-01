from inference import Inference
from model import Model
from model import RobertaClass

class App():
    ''' Model initilization and inferncing is done 
    '''
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        self.model = None 
        self.inference = None
        self.max_len = 256 
        self.device = 'cpu'   

    def start_model(self)->RobertaClass:
        self.model = Model() 
        # saved_model = RobertaClass()
        saved_model = self.model.get_model(self.model_file_path)
        tokenizer = self.model.get_model_tokenizer()
        self.inference = Inference(saved_model,tokenizer, self.max_len,self.device)
        return saved_model

    def predict(self,text):
        return self.inference.inference(text)
