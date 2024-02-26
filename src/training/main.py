from roberta import RobertaClass
import processor
from data import load_data, get_required_data
from model import BuildModel


device = processor.get_device()
model = RobertaClass()

LEARNING_RATE = 1e-05

MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16

df = load_data()
new_df = get_required_data(df)

model_build = BuildModel(model)
model_build.train(new_df, LEARNING_RATE,MAX_LEN, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE)

