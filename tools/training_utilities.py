import torch
from tqdm import tqdm


def acc_calc(labels, outputs):
    acc = labels == outputs
    acc = torch.reshape(acc, (-1,)).tolist()
    acc = acc.count(True) / len(acc)
    return acc


def train(model, train_data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    acc_total = 0.0
    for inputs, labels in tqdm(train_data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        labels, outputs = labels.to(int), torch.round(outputs).to(int)
        acc_total += acc_calc(labels, outputs)
        running_loss += loss.item()

    return running_loss / len(train_data_loader), acc_total / len(train_data_loader) * 100


def test(model, test_data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    acc_total = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(test_data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            labels, outputs = labels.to(int), torch.round(outputs).to(int)
            acc_total += acc_calc(labels, outputs)
            running_loss += loss.item()

    return running_loss / len(test_data_loader), acc_total / len(test_data_loader) * 100
