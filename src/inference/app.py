from inference import Inference
from model import Model
from roberta import RobertaClass

class App():
    ''' Model initilization and inferncing. Generic class for all Roberta models.
    '''
    def __init__(self, model_file_path="D:\Gimhan Sandeeptha\Gimhan\Sentiment-Email\ROBERTa_production\models\pytorch_roberta_sentiment_3_classes_0.1.3.bin"):
        self.model_file_path = model_file_path
        self.model = None 
        self.inference = None
        self.max_len = 256 
        self.device = 'cpu' 

    def start_model(self)->RobertaClass:
        '''' Load the saved model 
            After this is done model is redy for inferencing.
        '''
        self.model = Model()
        saved_model = self.model.get_model(self.model_file_path)
        tokenizer = self.model.get_model_tokenizer()
        self.inference = Inference(saved_model,tokenizer, self.max_len,self.device)
        return saved_model

    def predict(self,text):
        return self.inference.inference(text)
