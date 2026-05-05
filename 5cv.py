import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from dataset import MolDataset
from evaluate import evaluate
from model import GATNet
from seed import seed_everything

if __name__ == "__main__":
    SEED = 42
    dataset = MolDataset("train_hepatitis.csv")
    dims = dataset.get_feature_dims()


    config = {
        "hidden_dim": 32,
        "mlp_hidden": 256,
        "heads": 8,
        "mlp_layers": 2,
        "dropout": 0.3,
        "lr": 0.001,
        "weight_decay": 1e-5,
    }

    for key, value in config.items():
        print(f"  - {key}: {value}")
    print()

    seed_everything(SEED)
    labels = [int(d.y.item()) for d in dataset]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_results = []
    best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        train_subset = [dataset[i] for i in train_idx]
        val_subset = [dataset[i] for i in val_idx]

        seed_everything(SEED)
        g = torch.Generator().manual_seed(SEED)
        train_loader = DataLoader(
            train_subset, batch_size=32, shuffle=True,
            worker_init_fn=lambda wid: np.random.seed(SEED + wid),
            generator=g, num_workers=0
        )
        val_loader = DataLoader(
            val_subset, batch_size=32, shuffle=False,
            worker_init_fn=lambda wid: np.random.seed(SEED + wid),
            generator=g, num_workers=0
        )

        model = GATNet(
            input_dim=dataset[0].x.shape[1],
            hidden_dim=config["hidden_dim"],
            mlp_hidden=config["mlp_hidden"],
            heads=config["heads"],
            mlp_layers=config["mlp_layers"],
            dropout=config["dropout"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        loss_fn = nn.BCEWithLogitsLoss()

        best_score = -np.inf
        best_metrics = None
        best_epoch = 0

        for epoch in range(1, 101):
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

            metrics, _, _ = evaluate(val_loader, model, device)
            if metrics["AUC"] > best_score:
                best_score = metrics["AUC"]
                best_metrics = metrics
                best_epoch = epoch
                torch.save(model.state_dict(), f"best_model_fold{fold + 1}.pt")

            print(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                f"ACC={metrics['ACC']:.4f}, AUC={metrics['AUC']:.4f}, "
                f"SE={metrics['SE']:.4f}, SP={metrics['SP']:.4f}, F1={metrics['F1']:.4f}"
            )

        fold_results.append(best_metrics)
        best_epochs.append(best_epoch)
        print(f"Fold {fold + 1} best epoch: {best_epoch}")

    print("\n" + "=" * 60)
    for key in fold_results[0].keys():
        avg = np.mean([r[key] for r in fold_results])
        print(f"{key}: {avg:.4f}")

    avg_epoch = int(round(np.mean(best_epochs)))
    print("\nBest epochs per fold:", best_epochs)
    print("Average best epoch (rounded):", avg_epoch)
