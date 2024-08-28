import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
import math
import sys
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

n = sys.argv[1]
pt = sys.argv[2]

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_hyperparameters(filepath):
    hyperparameters = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if ':' in line and 'Best' not in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key in ['dim_model', 'dim_feedforward', 'nhead', 'num_layers']:
                    hyperparameters[key] = int(value)
                elif key == 'lr':
                    hyperparameters[key] = float(value)
    return hyperparameters

script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)

train_data_path = os.path.join(script_dir, f'dataset/{pt}/final_train_{n}.csv')
valid_data_path = os.path.join(script_dir, f'dataset/{pt}/final_val.csv')
test_data_path = os.path.join(script_dir, f'dataset/{pt}/final_test.csv')
besthp_path = os.path.join(script_dir, f'{pt}/results_{n}', 'besthp.txt')
test_prediction_path = os.path.join(script_dir, f'{pt}/results_{n}', 'test_prediction.csv')
val_prediction_path = os.path.join(script_dir, f'{pt}/results_{n}', 'val_prediction.csv')
test_results_path = os.path.join(script_dir, f'{pt}/results_{n}', 'test_results.txt')
val_results_path = os.path.join(script_dir, f'{pt}/results_{n}', 'val_results.txt')
model_path = os.path.join(script_dir, f'{pt}/results_{n}', 'best_model_grid_search.pth')

create_directory_if_not_exists(os.path.dirname(besthp_path))
create_directory_if_not_exists(os.path.dirname(test_results_path))
create_directory_if_not_exists(os.path.dirname(val_results_path))
create_directory_if_not_exists(os.path.dirname(test_prediction_path))
create_directory_if_not_exists(os.path.dirname(val_prediction_path))

class BattleDataset(Dataset):
    def __init__(self, dataframe, scaler=None):
        self.dataframe = dataframe
        if scaler is None:
            self.scaler = StandardScaler()
            self.sequences = self.create_sequences(dataframe, fit_scaler=True)
        else:
            self.scaler = scaler
            self.sequences = self.create_sequences(dataframe, fit_scaler=False)
        
    def create_sequences(self, dataframe, fit_scaler):
        sequences = []
        grouped = dataframe.groupby('ID')
        for _, group in grouped:
            group = group.sort_values('BattleNum')
            sequence = group.iloc[:, 2:-1].values
            label = group.iloc[0, -1]
            if fit_scaler:
                self.scaler.fit(sequence)
            sequence = self.scaler.transform(sequence)
            sequences.append((sequence, label))
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.float), torch.tensor(label, dtype=torch.long)

def collate_batch(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, labels

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dim_model, nhead, num_layers, dim_feedforward, max_len=5000, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.input_linear = nn.Linear(input_dim, dim_model)
        self.dim_model = dim_model
        self.positional_encoding = PositionalEncoding(dim_model, max_len=max_len)
        self.layers = nn.ModuleList([
            TransformerLayer(dim_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(dim_model, num_classes)

    def forward(self, src):
        src = self.input_linear(src)
        src = self.positional_encoding(src * math.sqrt(self.dim_model))
        for layer in self.layers:
            src = layer(src)
        output = self.output_layer(src.mean(dim=1))
        return output

class TransformerLayer(nn.Module):
    def __init__(self, dim_model, nhead, dim_feedforward, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src + self._sa_forward(src, src_mask, src_key_padding_mask))
        src = src2 + self._ff_forward(src2)
        src = self.norm2(src)
        return src

    def _sa_forward(self, src, src_mask, src_key_padding_mask):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        return self.dropout1(src2)

    def _ff_forward(self, src):
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.dropout2(src2)

def evaluate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    val_loss = 0
    ids, y_true, y_probs, y_pred = [], [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            predictions = (probs[:, 1] > threshold).long()
            y_true.extend(y.tolist())
            y_probs.extend(probs[:, 1].tolist()) 
            y_pred.extend(predictions.tolist())
    val_loss /= len(dataloader)
    return y_true, y_pred, y_probs, val_loss

def main():
    train_df = pd.read_csv(train_data_path)
    valid_df = pd.read_csv(valid_data_path)
    test_df = pd.read_csv(test_data_path)
    
    train_dataset = BattleDataset(train_df)
    valid_dataset = BattleDataset(valid_df, scaler=train_dataset.scaler)
    test_dataset = BattleDataset(test_df, scaler=train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    input_dim = train_dataset[0][0].shape[1]
    
    # 读取besthp.txt中的超参数
    hyperparameters = load_hyperparameters(besthp_path)

    model = TransformerClassifier(
        input_dim=input_dim, 
        num_classes=2, 
        dim_model=hyperparameters['dim_model'], 
        dim_feedforward=hyperparameters['dim_feedforward'], 
        nhead=hyperparameters['nhead'], 
        num_layers=hyperparameters['num_layers']
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(model_path))
    
    # 测试集预测与评估
    y_true_test, y_pred_test, y_probs_test, test_loss = evaluate(model, test_loader, criterion, device, threshold=0.5)
    test_results_df = pd.DataFrame({
        'True Label': y_true_test,
        'Prediction': y_pred_test,
        'Pred_Score': y_probs_test,
    })
    test_results_df.to_csv(test_prediction_path, index=False)

    accuracy_test = accuracy_score(y_true_test, y_pred_test)
    precision_test = precision_score(y_true_test, y_pred_test)
    recall_test = recall_score(y_true_test, y_pred_test)
    f1_test = f1_score(y_true_test, y_pred_test)

    with open(test_results_path, 'w') as f:
        f.write(f'Test Results\n')
        f.write(f'Accuracy: {accuracy_test}\n')
        f.write(f'Precision: {precision_test}\n')
        f.write(f'Recall: {recall_test}\n')
        f.write(f'F1 Score: {f1_test}\n')

    # 验证集预测与评估
    y_true_val, y_pred_val, y_probs_val, val_loss = evaluate(model, valid_loader, criterion, device, threshold=0.5)
    val_results_df = pd.DataFrame({
        'True Label': y_true_val,
        'Prediction': y_pred_val,
        'Pred_Score': y_probs_val,
    })
    val_results_df.to_csv(val_prediction_path, index=False)

    accuracy_val = accuracy_score(y_true_val, y_pred_val)
    precision_val = precision_score(y_true_val, y_pred_val)
    recall_val = recall_score(y_true_val, y_pred_val)
    f1_val = f1_score(y_true_val, y_pred_val)

    with open(val_results_path, 'w') as f:
        f.write(f'Validation Results\n')
        f.write(f'Accuracy: {accuracy_val}\n')
        f.write(f'Precision: {precision_val}\n')
        f.write(f'Recall: {recall_val}\n')
        f.write(f'F1 Score: {f1_val}\n')

    print("Testing and validation complete.")

if __name__ == '__main__':
    main()
