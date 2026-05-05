from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import torch


def evaluate(loader, model, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = torch.sigmoid(out)
            all_labels.append(batch.y.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    pred_labels = (all_preds > 0.5).int()

    acc = accuracy_score(all_labels, pred_labels)
    auc_score = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, pred_labels)
    tn, fp, fn, tp = confusion_matrix(all_labels, pred_labels).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)

    metrics = {"ACC": acc, "AUC": auc_score, "SE": se, "SP": sp, "F1": f1}
    return metrics, all_labels.numpy(), all_preds.numpy()