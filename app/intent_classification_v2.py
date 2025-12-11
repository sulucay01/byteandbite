import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# =========================
# Config
# =========================

@dataclass
class Config:
    data_path: str = "/restaurant_intent_data.jsonl"
    model_name: str = "bert-base-uncased"
    max_length: int = 64
    num_epochs: int = 5
    early_stopping_patience: int = 2
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # will be overridden per run
    learning_rate: float = 2e-5
    batch_size: int = 16
    weight_decay: float = 0.01


cfg = Config()


# =========================
# Utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: str):
    texts, labels, ids = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ids.append(obj.get("id"))
            texts.append(obj["text"])
            labels.append(obj["labels"])
    return texts, labels, ids


def build_label_mapping(all_label_lists: List[List[str]]):
    label_set = set()
    for labs in all_label_lists:
        label_set.update(labs)
    sorted_labels = sorted(label_set)
    label2id = {l: i for i, l in enumerate(sorted_labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def multilabel_to_vector(labels: List[str], label2id: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(label2id), dtype=np.float32)
    for l in labels:
        if l in label2id:
            vec[label2id[l]] = 1.0
    return vec


class IntentDataset(Dataset):
    def __init__(self, texts, label_vectors, tokenizer, max_length):
        self.texts = texts
        self.labels = label_vectors
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


def create_dataloaders(texts, label_lists, label2id, tokenizer, cfg_run: Config):
    label_vectors = np.stack(
        [multilabel_to_vector(labs, label2id) for labs in label_lists]
    )

    test_val_size = cfg_run.val_size + cfg_run.test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts,
        label_vectors,
        test_size=test_val_size,
        random_state=cfg_run.seed,
        shuffle=True,
    )

    val_ratio_of_temp = cfg_run.val_size / test_val_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1.0 - val_ratio_of_temp,
        random_state=cfg_run.seed,
        shuffle=True,
    )

    train_ds = IntentDataset(X_train, y_train, tokenizer, cfg_run.max_length)
    val_ds = IntentDataset(X_val, y_val, tokenizer, cfg_run.max_length)
    test_ds = IntentDataset(X_test, y_test, tokenizer, cfg_run.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=cfg_run.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg_run.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg_run.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


# =========================
# Train / Eval
# =========================

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return total_loss / len(dataloader), f1_micro, f1_macro


# =========================
# Single experiment run
# =========================

def run_experiment(lr, batch_size, weight_decay) -> Dict[str, Any]:
    cfg_run = Config(
        learning_rate=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
    )

    set_seed(cfg_run.seed)

    texts, label_lists, _ = load_dataset(cfg_run.data_path)
    label2id, id2label = build_label_mapping(label_lists)

    tokenizer = BertTokenizerFast.from_pretrained(cfg_run.model_name)
    train_loader, val_loader, _ = create_dataloaders(
        texts, label_lists, label2id, tokenizer, cfg_run
    )

    num_labels = len(label2id)
    model = BertForSequenceClassification.from_pretrained(
        cfg_run.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )
    model.to(cfg_run.device)

    total_steps = len(train_loader) * cfg_run.num_epochs
    optimizer = AdamW(
        model.parameters(),
        lr=cfg_run.learning_rate,
        weight_decay=cfg_run.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, cfg_run.num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, cfg_run.device
        )
        val_loss, val_f1_micro, val_f1_macro = evaluate(
            model, val_loader, cfg_run.device
        )

        print(
            f"  Epoch {epoch}/{cfg_run.num_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val F1-micro: {val_f1_micro:.4f} | Val F1-macro: {val_f1_macro:.4f}"
        )

        if val_f1_micro > best_val_f1:
            best_val_f1 = val_f1_micro
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg_run.early_stopping_patience:
                break

    return {
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "best_val_f1_micro": best_val_f1,
        "best_epoch": best_epoch,
    }


# =========================
# Main: lightweight grid search
# =========================

def main():
    # small grid
    lrs = [1e-5, 2e-5, 3e-5]
    batch_sizes = [16, 32]
    weight_decays = [0.0, 0.01]

    results = []

    for lr in lrs:
        for bs in batch_sizes:
            for wd in weight_decays:
                print("\n========================================")
                print(f"Running experiment: lr={lr}, batch_size={bs}, weight_decay={wd}")
                print("========================================")
                res = run_experiment(lr, bs, wd)
                results.append(res)
                print(
                    f"--> DONE | best_val_f1_micro={res['best_val_f1_micro']:.4f} "
                    f"at epoch {res['best_epoch']}"
                )

    # sort by best_val_f1_micro descending
    results = sorted(results, key=lambda x: x["best_val_f1_micro"], reverse=True)

    print("\n===== SUMMARY (sorted by best val F1-micro) =====")
    for r in results:
        print(
            f"lr={r['lr']:<8} bs={r['batch_size']:<2} wd={r['weight_decay']:<4} "
            f"| best_val_f1_micro={r['best_val_f1_micro']:.4f} "
            f"| best_epoch={r['best_epoch']}"
        )


if __name__ == "__main__":
    main()
