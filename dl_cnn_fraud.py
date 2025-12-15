from .db_utils import read_sql_df
"""
CNN for document fraud detection:
- Simple PyTorch CNN skeleton for binary classification on document images
"""
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from .logging_utils import get_logger
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score
import torch.nn.functional as F

logger = get_logger("cnn_fraud")

class DocDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        label = 1 if "fraud" in path.stem.lower() else 0  # placeholder rule
        if self.transform: img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 5
    img_size: Tuple[int, int] = (224, 224)

def load_doc_storage_uri():
    """
    Loads doc_storage_uri values from the MySQL 'document' table using db_utils.read_sql_df.
    Returns: List of doc_storage_uri strings.
    """
    df = read_sql_df("SELECT doc_storage_uri FROM document")
    return df['doc_storage_uri'].tolist()
    
def train_cnn(cfg: TrainConfig = TrainConfig()):
    transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    doc_paths = [Path(uri) for uri in load_doc_storage_uri()]
    ds = DocDataset(doc_paths, transform)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    model = SimpleCNN()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in dl:
            opt.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)
        logger.info(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}")

    # --- Evaluation & Visualizations ---
    evaluate_cnn_model(model, dl)
    return model

# --- Evaluation & Visualization Utilities ---
def evaluate_cnn_model(model, dl, threshold=0.5):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x)
            probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    plt.title(f"Confusion Matrix (threshold={threshold})")
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud","Fraud"], yticklabels=["Not Fraud","Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    st.subheader("Confusion Matrix")
    st.pyplot(plt.gcf())
    plt.clf()

    # 2. ROC Curve & AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    st.subheader("ROC Curve & AUC")
    st.pyplot(plt.gcf())
    plt.clf()

    # 3. Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    st.subheader("Precision-Recall Curve")
    st.pyplot(plt.gcf())
    plt.clf()

    # 4. Threshold Analysis Plot (Precision, Recall, F1 vs threshold)
    f1s = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, preds))
    plt.figure()
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.plot(thresholds, f1s, label='F1')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Analysis')
    plt.legend()
    plt.tight_layout()
    st.subheader("Threshold Analysis Plot")
    st.pyplot(plt.gcf())
    plt.clf()

    # 5. Lift/Gain Chart
    st.subheader("Lift/Gain Chart")
    plot_lift_gain(y_true, y_prob)



def plot_lift_gain(y_true, y_prob, n_bins=10):
    import matplotlib.pyplot as plt
    import streamlit as st
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]
    total_positives = y_true.sum()
    bins = np.array_split(y_true, n_bins)
    gains = np.cumsum([b.sum() for b in bins]) / total_positives
    lift = gains / (np.arange(1, n_bins+1)/n_bins)
    plt.figure()
    plt.plot(np.arange(1, n_bins+1)/n_bins, gains, label='Gain')
    plt.plot(np.arange(1, n_bins+1)/n_bins, lift, label='Lift')
    plt.xlabel('Fraction of Sample')
    plt.ylabel('Gain / Lift')
    plt.title('Lift & Gain Chart')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

