import torch
from tqdm import tqdm
import datetime
import os

class RobertaFinetune():
    def __init__(self, model, optimizer, loss_function, early_stopping_patience=3) -> None:
        # Creating the loss function and optimizer
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.early_stopping_patience = early_stopping_patience
        self.min_val_loss = float('inf')
        self.best_model = None
        self.no_improvement_count = 0
        self.log = Log()

    def calculate_accuracy(self,preds, targets):
        n_correct = (preds==targets).sum().item()
        return n_correct

    # Validation Function
    def validate(self, validation_loader, device):
        self.model.eval()
        val_loss = 0
        n_correct = 0
        nb_val_steps = 0
        nb_val_examples = 0

        with torch.no_grad():
            for _, data in tqdm(enumerate(validation_loader, 0)):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)

                outputs = self.model(ids, mask, token_type_ids)
                loss = self.loss_function(outputs, targets)
                val_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim=1)
                n_correct += self.calculate_accuracy(big_idx, targets)

                nb_val_steps += 1
                nb_val_examples += targets.size(0)

        val_accuracy = (n_correct * 100) / nb_val_examples
        val_loss /= nb_val_steps

        # print(f'Validation Loss: {val_loss}')
        # print(f'Validation Accuracy: {val_accuracy}%')
        # print()
        log_dict={
            "Validation Loss": val_loss,
            "Validation Accuracy": f"{val_accuracy}%"
        }
        self.log.enter_log(log_dict)

        if val_loss < self.min_val_loss:
            self.min_val_loss=val_loss
            self.best_model = self.model.state_dict()
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        return val_loss
    
    def save_checkpoint(self, filepath):
        # Remove existing .pth files if they exist
        existing_files = [f for f in os.listdir(os.path.dirname(filepath)) if f.endswith('.pth')]
        for existing_file in existing_files:
            os.remove(os.path.join(os.path.dirname(filepath), existing_file))

        checkpoint = {
            # After finetuning self.model is either bestmodel with validation is not None, if not validate it is the model tuned for a indicated number of epochs. 
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        # print(f"Checkpoint saved at {filepath}")

    # Training loop
    def finetune(self, training_loader, validation_loader=None, device='cpu', epochs=3, checkpoint_folder = "models/finetune_checkpoints"):
        current_date_time = datetime.datetime.now()
        formatted_date_time = current_date_time.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{formatted_date_time}.pth"
        checkpoint_file_path=os.path.join(checkpoint_folder,file_name)
        # print(checkpoint_file_path)
        
        for epoch in range(epochs):
            tr_loss = 0
            n_correct = 0
            nb_tr_steps = 0
            nb_tr_examples = 0
            self.model.train()

            for _, data in tqdm(enumerate(training_loader, 0)):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)

                outputs = self.model(ids, mask, token_type_ids)
                loss = self.loss_function(outputs, targets)
                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim=1)
                n_correct += self.calculate_accuracy(big_idx, targets)

                nb_tr_steps += 1
                nb_tr_examples += targets.size(0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # print()
            # print(f'Epoch {epoch + 1}')
            # print(f'Training Loss: {tr_loss / nb_tr_steps}')
            # print(f'Training Accuracy: {(n_correct * 100) / nb_tr_examples}%')
            log_dict = {"Epoch": {epoch + 1}, 
                        "Training Loss": tr_loss / nb_tr_steps, 
                        "Training Accuracy": f"{(n_correct * 100) / nb_tr_examples}%"
                        }
            self.log.enter_log(log_dict)

            # Validation
            if validation_loader is not None:
                self.validate(validation_loader, device)

                if self.no_improvement_count >= self.early_stopping_patience:
                    print("Early stopping triggered!")
                    break
    
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)

        self.save_checkpoint(checkpoint_file_path)
        return self.model
    
class Log():
    def __init__(self) -> None:
        self.max_file_count=20

        self.log_folder_path= "logs"
        current_datetime = datetime.datetime.now()
        filename = current_datetime.strftime("%Y-%m-%d_%H-%M-%S.txt")
        self.file_path = os.path.join(self.log_folder_path, filename)

    def enter_log(self, log:dict):
        with open(self.file_path,'a') as file: #Append mode
            for key, value in log.items():
                file.write(f"{key}: {value}\n" )
        file_list = os.listdir(self.log_folder_path)
        num_files=len(file_list)

        if num_files>self.max_file_count:
            self.delete_oldest_file()

    def delete_oldest_file(self):
        files = os.listdir(self.log_folder_path)
        files = [f for f in files if os.path.isfile(os.path.join(self.log_folder_path, f))] # Filter out directories (if any)

        if files:
            file_times = [(f, os.path.getctime(os.path.join(self.log_folder_path, f))) for f in files] #Get the metadata change of each file
            sorted_files = sorted(file_times, key=lambda x: x[1])

            # Get the oldest file
            oldest_file_name = sorted_files[0][0]
            file_path = os.path.join(self.log_folder_path, oldest_file_name)

            # Delete the oldest file
            os.remove(file_path)
            print("Oldest file deleted:", oldest_file_name)
        else:
            print("No files in the directory to delete.")


# # Unit test for Log class
# log = Log()
# dict = {"val_loss": 0.1457, "val_accuracy":0.89, "train_loss":0.2578, "train_accuracy":0.75}    
# dict2 = {"val_loss": 0.1457, "val_accuracy":0.89, "train_loss":0.2578, "train_accuracy":0.75} 
# log.enter_log(dict)
# log.enter_log(dict2)
# # log.delete_oldest_file()
