from ..model.roberta import RobertaClass
from ..train.processor import get_device
from ..train.model import BuildModel
from ..train.data import data
import torch

device = get_device()
model = RobertaClass(hidden_size=768,dropout_prob=0.3, num_classes=3)

LEARNING_RATE = 1e-05
MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16

df = data.load_data()
new_df = data.get_required_data(df)

model_build = BuildModel(model)

loss_function = torch.nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
model_build.train(new_df,MAX_LEN, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, optimizer, loss_function)
