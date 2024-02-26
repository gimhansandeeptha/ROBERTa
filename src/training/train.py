import torch
from tqdm import tqdm

def calculate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

# Validation function
def validate(model, validation_loader, loss_function, device):
    model.eval()
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

            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs, targets)
            val_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accuracy(big_idx, targets)

            nb_val_steps += 1
            nb_val_examples += targets.size(0)

    val_accuracy = (n_correct * 100) / nb_val_examples
    val_loss /= nb_val_steps

    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_accuracy}%')
    print()

    return

# Training loop
def train(model, training_loader, validation_loader, loss_function, optimizer, device, epochs=5):
    for epoch in range(epochs):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        model.train()

        for _, data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print()
        print(f'Training Epoch {epoch + 1}')
        print(f'Training Loss: {tr_loss / nb_tr_steps}')
        print(f'Training Accuracy: {(n_correct * 100) / nb_tr_examples}%')

        # Validation
        val_accuracy = validate(model, validation_loader, loss_function, device)
    return