import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn.utils.rnn import pad_sequence
import math
import sys
import os
from itertools import product

n = sys.argv[1]
pt = sys.argv[2]

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)

train_data_path = os.path.join(script_dir, f'dataset/{pt}/final_train_{n}.csv')
valid_data_path = os.path.join(script_dir, f'dataset/{pt}/final_val.csv')
test_data_path = os.path.join(script_dir, f'dataset/{pt}/final_test.csv')
besthp_path = os.path.join(script_dir, f'{pt}/results_{n}', 'besthp.txt')
results_path = os.path.join(script_dir, f'{pt}/results_{n}', 'model_performance_results.txt')
create_directory_if_not_exists(os.path.dirname(besthp_path))
create_directory_if_not_exists(os.path.dirname(results_path))

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

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            val_loss += loss.item()
            predictions = output.argmax(dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(predictions.tolist())
    val_loss /= len(dataloader)
    return y_true, y_pred, val_loss

def grid_search_hyperparameters(epochs, train_loader, valid_loader, input_dim, device):
    best_valid_acc = 0
    best_score = 0
    best_hyperparameters = {}
    best_epoch = 0

    dim_models = [32, 64, 128, 256, 512]
    dim_feedforwards = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    nheads = [1, 2, 4]
    num_layers_list = [1, 2, 3]
    lrs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]

    all_combinations = list(product(dim_models, dim_feedforwards, nheads, num_layers_list, lrs))

    for combination in all_combinations:
        dim_model, dim_feedforward, nhead, num_layers, lr = combination
        model = TransformerClassifier(input_dim=input_dim, num_classes=2, dim_model=dim_model, dim_feedforward = dim_feedforward, nhead=nhead, num_layers=num_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print(f"Trial: dim_model={dim_model}, dim_feedforward={dim_feedforward} nhead={nhead}, num_layers={num_layers}, lr={lr}")
        val_loss_min = float('inf')
        val_loss_counter = 0

        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            y_true, y_pred, val_loss = evaluate(model, valid_loader, criterion, device)
            accuracy = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            score = recall + accuracy

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                val_loss_counter = 0
                if score > best_score and recall >= 0.60 and accuracy >= 0.61:
                    best_score = score
                    best_hyperparameters = {
                        'dim_model': dim_model,
                        'dim_feedforward': dim_feedforward,
                        'nhead': nhead,
                        'num_layers': num_layers,
                        'lr': lr
                    }
                    best_epoch = epoch + 1
                    model_path = os.path.dirname(results_path)
                    create_directory_if_not_exists(model_path)
                    
                    model_save_path = os.path.join(model_path, 'best_model_grid_search.pth')
                    torch.save(model.state_dict(), model_save_path)
                    
                    performance_info_path = os.path.join(model_path, 'best_performance_info.txt')
                    with open(performance_info_path, 'w') as file:
                        file.write(f"Best Model saved at: {model_save_path}\n")
                        file.write(f"Best Epoch: {best_epoch}\n")
                        file.write(f"Hyperparameters:\n")
                        for param, value in best_hyperparameters.items():
                            file.write(f"{param}: {value}\n")
                        file.write(f"Recall: {recall:.4f}\n")
                        file.write(f"Accuracy: {accuracy:.4f}\n")
                        file.write(f"Score: {score:.4f}\n")

                    print(f"New best performance on epoch {best_epoch} with {best_hyperparameters}, recall: {recall}, acc: {accuracy}")

            else:
                val_loss_counter += 1
                if val_loss_counter >= 50:
                    print("Early stopping triggered.")
                    break

    print(f"Best performance, recall: {recall}, acc: {accuracy} with Hyperparameters: {best_hyperparameters} at Epoch {best_epoch}")
    
    return best_hyperparameters, recall, accuracy

def main():
    train_df = pd.read_csv(train_data_path)
    valid_df = pd.read_csv(valid_data_path)
    test_df = pd.read_csv(test_data_path)
    
    train_dataset = BattleDataset(train_df)
    # valid_dataset = BattleDataset(valid_df)
    # test_dataset = BattleDataset(test_df)
    valid_dataset = BattleDataset(valid_df, scaler=train_dataset.scaler)
    test_dataset = BattleDataset(test_df, scaler=train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    input_dim = train_dataset[0][0].shape[1]
    # model = TransformerClassifier(input_dim=input_dim, num_classes=2).to(device)
    # model = TransformerClassifier(input_dim=input_dim, num_classes=2, dim_model=12).to(device)
    # model = TransformerClassifier(input_dim=input_dim, num_classes=2, dim_model=12, dim_feedforward=32, nhead=2, num_layers=1).to(device)



    best_hyperparameters, recall, accuracy = grid_search_hyperparameters(
        epochs=300,
        train_loader=train_loader,
        valid_loader=valid_loader,
        input_dim=input_dim,
        device=device
    )

    with open(besthp_path, 'w') as f:
        f.write(f"Best Validation Recall: {recall}, Acc: {accuracy}\n")
        f.write("Best Hyperparameters:\n")
        for param, value in best_hyperparameters.items():
            f.write(f"{param}: {value}\n")
    
    model = TransformerClassifier(
        input_dim=input_dim, 
        dim_feedforward = best_hyperparameters['dim_feedforward'],
        num_classes=2, 
        dim_model=best_hyperparameters['dim_model'], 
        nhead=best_hyperparameters['nhead'], 
        num_layers=best_hyperparameters['num_layers']
    ).to(device)




    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    optimizer = optim.Adam(model.parameters(), lr=best_hyperparameters['lr'])
    criterion = nn.CrossEntropyLoss()

    best_valid_acc = 0
    best_valid_score = 0
    best_valid_recall = 0
    best_valid_precision = 0
    best_valid_f1 = 0
    best_epoch = 0
    early_stopping_patience = 100
    early_stopping_counter = 0
    previous_best_loss = 1

    epochs = 600
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        y_true, y_pred, val_loss = evaluate(model, valid_loader, criterion, device)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        score = recall + accuracy

        if val_loss > previous_best_loss:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
        else:
            early_stopping_counter = 0
            previous_best_loss = val_loss
        
        if score > best_valid_score and recall > 0.6 and accuracy > 0.61:
            best_valid_score = score
            best_valid_acc = accuracy
            best_valid_recall = recall
            best_valid_precision = precision
            best_valid_f1 = f1
            best_epoch = epoch + 1
            print(f'update model on epoch {best_epoch}')
            # print(score)
            torch.save(model.state_dict(), os.path.join(os.path.dirname(results_path), 'best_model.pth'))

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Valid Loss: {val_loss}, Valid Recall: {recall}, Precision: {precision}, Acc: {accuracy}')
        # scheduler.step(train_loss)

    model.load_state_dict(torch.load(os.path.join(os.path.dirname(results_path), 'best_model.pth')))
    y_true_test, y_pred_test, test_loss = evaluate(model, test_loader, criterion, device)
    accuracy_test = accuracy_score(y_true_test, y_pred_test)
    precision_test = precision_score(y_true_test, y_pred_test)
    recall_test = recall_score(y_true_test, y_pred_test)
    f1_test = f1_score(y_true_test, y_pred_test)

    with open(results_path, 'w') as f:
        f.write(f'Best Validation Performance (Epoch {best_epoch})\n')
        f.write(f'Best Valid Accuracy: {best_valid_acc}\n')
        f.write(f'Best Valid Recall: {best_valid_recall}\n')
        f.write(f'Best Valid Precision: {best_valid_precision}\n')
        f.write(f'Best Valid F1 Score: {best_valid_f1}\n\n')

        f.write(f'Test Results\n')
        f.write(f'Accuracy: {accuracy_test}\n')
        f.write(f'Precision: {precision_test}\n')
        f.write(f'Recall: {recall_test}\n')
        f.write(f'F1 Score: {f1_test}\n')

    print("Training complete. Best validation and test results saved to", results_path)

if __name__ == '__main__':
    main()
