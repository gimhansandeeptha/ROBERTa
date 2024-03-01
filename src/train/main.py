# Your script in directory A
import sys
sys.path.append('../data')  # Adjust the path accordingly

# Now you can import modules from the sibling directory
from data_module import data

from roberta import RobertaClass
import processor
from data_module import data
from model import BuildModel
import torch

device = processor.get_device()
model = RobertaClass(hidden_size=768,dropout_prob=0.3, num_classes=3)

LEARNING_RATE = 1e-05
MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16

df = data.load_data()
new_df = data.get_required_data(df)

model_build = BuildModel(model)

optimizer= torch.nn.CrossEntropyLoss()
loss_function=torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
model_build.train(new_df, LEARNING_RATE,MAX_LEN, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE)
