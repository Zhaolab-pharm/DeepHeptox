from dataset import MolDataset
from evaluate import evaluate
from model import GATNet
from seed import seed_everything

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
def train_and_test(
        train_csv_path,
        test_csv_path,
        num_epochs=50,
        fixed_epoch=48,
        batch_size=64,
        seed=42,
        hidden_dim=64,
        mlp_hidden=256,
        heads=8,
        mlp_layers=2,
        dropout=0.3,
        lr=0.001,
        weight_decay=1e-5
):
    train_dataset = MolDataset(train_csv_path)
    test_dataset = MolDataset(test_csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=lambda wid: np.random.seed(seed + wid),
        generator=g,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=lambda wid: np.random.seed(seed + wid),
        generator=g,
        num_workers=0
    )

    model = GATNet(
        input_dim=train_dataset[0].x.shape[1],
        hidden_dim=hidden_dim,
        mlp_hidden=mlp_hidden,
        heads=heads,
        mlp_layers=mlp_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        train_loss = total_loss / len(train_loader.dataset)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        # 固定epoch保存模型
        if epoch == fixed_epoch:
            torch.save(model.state_dict(), "model.pt")

    model.load_state_dict(torch.load("model.pt"))
    metrics, labels_np, preds_np = evaluate(test_loader, model, device)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    return metrics


if __name__ == "__main__":
    SEED = 42
    seed_everything(SEED)
    config = {
        "hidden_dim": 32,
        "mlp_hidden": 256,
        "heads": 8,
        "mlp_layers": 2,
        "dropout": 0.3,
        "lr": 0.001,
        "weight_decay": 1e-5
    }

    FIXED_EPOCH = 100
    final_metrics = train_and_test(
        train_csv_path="train_hepatitis.csv",
        test_csv_path="test_hepatitis.csv",
        num_epochs=100,
        fixed_epoch=FIXED_EPOCH,
        batch_size=32,
        seed=SEED,
        **config
    )