"""
Train the deep metric learning model for one-shot DB configuration transfer.

Usage:
    cd scripts && python train_dml_model.py \
        --triplet_csv ../DBMSTransferLearning/dataset/full_triplet_data_concordance.csv \
        --context_csv ../DBMSTransferLearning/dataset/context_default_metrics_all.csv \
        --output_dir ../autotune/optimizer/dml_models/

Outputs:
    - context_model.pth (model weights)
    - scaler.pkl (fitted MinMaxScaler on context default metrics)
    - context_default_metrics_all.csv (copied for reference)
"""
import argparse
import os
import shutil
import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


INPUT_DIM = 11
FEATURE_COLS = [
    'Average Memory Usage Percentage', 'InnoDB Buffer Pool Cache Hit Rate',
    'InnoDB Dirty Buffer Pages', 'Current QPS (Queries Per Second)',
    'Max CPU Usage (100 - Idle)', 'InnoDB Rows Deleted (60s Rate)',
    'InnoDB Rows Inserted (60s Rate)', 'InnoDB Rows Read (60s Rate)',
    'InnoDB Rows Updated (60s Rate)', 'Average Disk IOPS (Read)',
    'Average Disk IOPS (Write)'
]


class TripletDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values.astype('float32')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        d = INPUT_DIM
        anchor = row[0:d]
        pos = row[d:2*d]
        neg = row[2*d:3*d]
        return torch.tensor(anchor), torch.tensor(pos), torch.tensor(neg)


class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

    def forward(self, x):
        output = self.net(x)
        return nn.functional.normalize(output, p=2, dim=1)


def train_model(triplet_csv, embedding_dim=16, batch_size=64, lr=0.001, epochs=50, margin=1.0):
    df = pd.read_csv(triplet_csv)
    df = df.drop(columns=['anchor_id', 'pos_id', 'neg_id'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(TripletDataset(df), batch_size=batch_size, shuffle=True)
    model = EmbeddingNet(INPUT_DIM, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for anchor, pos, neg in train_loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()
            emb_a = model(anchor)
            emb_p = model(pos)
            emb_n = model(neg)
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train DML model for DB config transfer')
    parser.add_argument('--triplet_csv', default='../DBMSTransferLearning/dataset/full_triplet_data_concordance.csv')
    parser.add_argument('--context_csv', default='../DBMSTransferLearning/dataset/context_default_metrics_all.csv')
    parser.add_argument('--output_dir', default='../autotune/optimizer/dml_models/')
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--margin', type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Train model
    print(f"Training on {args.triplet_csv}")
    model = train_model(args.triplet_csv, args.embedding_dim, args.batch_size, args.lr, args.epochs, args.margin)

    # Save model
    model_path = os.path.join(args.output_dir, 'context_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Fit and save scaler on context default metrics
    context_df = pd.read_csv(args.context_csv)
    scaler = MinMaxScaler()
    scaler.fit(context_df[FEATURE_COLS])
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Copy context metrics CSV
    dest_csv = os.path.join(args.output_dir, 'context_default_metrics_all.csv')
    shutil.copy2(args.context_csv, dest_csv)
    print(f"Context metrics copied to {dest_csv}")

    print("Done! Model artifacts saved to", args.output_dir)


if __name__ == "__main__":
    main()
