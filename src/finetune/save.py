import torch
# saved_model_path = "C:\\Users\\gimhanSandeeptha\\Gimhan Sandeeptha\\Sentiment Project\\ROBERTa\\models\\pytorch_roberta_sentiment_3_classes_0.1.3.bin"
# loaded_model = torch.load(saved_model_path,map_location=torch.device('cpu'))
# _PATH = "C:\\Users\\gimhanSandeeptha\\Gimhan Sandeeptha\\Sentiment Project\\ROBERTa\\models\\state_dict.pth"

def save_model_dict(model, path):
    ''' Given the model and path to save save the model state dict in the indicated location 
    '''
    torch.save({
                'epoch': 5,
                'model_state_dict': model.state_dict(),
                'loss': 2.134,
                }, path)

def load_model(model, path):
    '''Load the initialized model with the state dict in the path
    '''
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_data():
    return "{The data goes here}"

def finetune(model,):
    model
    