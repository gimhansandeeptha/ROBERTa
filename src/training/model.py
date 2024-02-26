import torch
from torch.utils.data import DataLoader
from training import load_data, get_required_data, split
from transformers import RobertaModel, RobertaTokenizer

class SentimentData():
    '''
    Custom class for handling sentiment data. To be able to pass to the DataLoader.
    __getitem__ function do the tokenization for each text sample.

    '''
    def __init__(self, x, y, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = x
        self.targets = y
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class RobertaClass(torch.nn.Module):
  '''
  Custom PyTorch module for sentiment analysis using a fine-tuned RoBERTa model.
  - l1: Pre-trained RoBERTa model loaded from "roberta-base" using Hugging Face Transformers.
  - pre_classifier: Linear layer for additional transformation before classification.
  - dropout: Dropout layer for regularization.
  - classifier: Linear layer for final sentiment classification.
  '''
  def __init__(self):
      super(RobertaClass, self).__init__()
      self.l1 = RobertaModel.from_pretrained("roberta-base")
      self.pre_classifier = torch.nn.Linear(768, 768)
      self.dropout = torch.nn.Dropout(0.3)
      self.classifier = torch.nn.Linear(768, 3)

  def forward(self, input_ids, attention_mask, token_type_ids):
      output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
      hidden_state = output_1[0]
      pooler = hidden_state[:, 0]
      pooler = self.pre_classifier(pooler)
      pooler = torch.nn.ReLU()(pooler)
      pooler = self.dropout(pooler)
      output = self.classifier(pooler)
      return output
  

MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
# EPOCHS = 1
LEARNING_RATE = 1e-05

# Use the pretrained Roberta Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

df = load_data()
new_df = get_required_data(df)

x_train, y_train, x_test, y_test, x_val, y_val = split(new_df)

print(f"{x_train[:3]}\n {y_train[:3]} \n {x_test[:3]} \n {y_test[:3]} \n {x_val[:3]} \n {y_val[:3]}")

training_set = SentimentData(x_train, y_train, tokenizer, MAX_LEN)
testing_set = SentimentData(x_train, y_train, tokenizer, MAX_LEN)
validation_set = SentimentData(x_val, y_val, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': 1,
                'shuffle': False,
                'num_workers': 0
                }

validation_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
validation_loader = DataLoader(validation_set, **validation_params)


# Iterate over a few batches from the training loader and print samples
for batch in training_loader:
    print("Batch Sample:")
    print("Input IDs:", batch['ids'])
    print("Attention Mask:", batch['mask'])
    print("Token Type IDs:", batch['token_type_ids'])
    print("Targets:", batch['targets'])
    break  # Break after the first batch to view a few samples