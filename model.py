import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class VehicularGRU_IDS(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=2, num_classes=2):
        super(VehicularGRU_IDS, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
       
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def preprocess_ciciov_data(filepath, target_label, sequence_length=10):
    """Loads a chosen decimal CSV and structures sliding temporal windows."""
    df = pd.read_csv(filepath)
   
    # Target the 9 core features available in the CICIoV2024 decimal logs
    feature_cols = ['ID', 'DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7']
    df[feature_cols] = df[feature_cols].fillna(0)
    features = df[feature_cols].values
   
    # Establish labels based on whether this client is running normal or attack logs
    labels = np.full(len(features), target_label, dtype=np.int64)
   
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
   
    X_seq, y_seq = [], []
    for i in range(len(features) - sequence_length):
        X_seq.append(features[i:i+sequence_length])
        y_seq.append(labels[i+sequence_length - 1])
       
    return torch.tensor(np.array(X_seq), dtype=torch.float32), torch.tensor(np.array(y_seq), dtype=torch.long)

def train(net, trainloader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    net.train()
    for _ in range(epochs):
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    return loss / len(testloader) if len(testloader) > 0 else 0.0, accuracy
