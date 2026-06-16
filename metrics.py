import numpy as np
import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, device, criterion=None):
    criterion = criterion or nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        batch_size = yb.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == yb).sum().item()
        total_samples += batch_size

    average_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return average_loss, accuracy


@torch.no_grad()
def evaluate_model(model, loader, device, criterion=None):
    criterion = criterion or nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    targets = []
    predictions = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        batch_size = yb.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == yb).sum().item()
        total_samples += batch_size
        targets.extend(yb.cpu().tolist())
        predictions.extend(preds.cpu().tolist())

    average_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "targets": targets,
        "predictions": predictions,
    }


def build_confusion_matrix(targets, predictions, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return matrix


def normalize_confusion_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    row_sums = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    return matrix / row_sums
