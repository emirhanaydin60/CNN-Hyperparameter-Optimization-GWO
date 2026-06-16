import os

import matplotlib.pyplot as plt
import numpy as np

from metrics import normalize_confusion_matrix


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_global(global_bests, out_path):
    _ensure_parent_dir(out_path)
    plt.figure(figsize=(8, 4))
    plt.plot(global_bests, marker="o", label="Global Best")
    plt.title("GWO Global Best Fitness")
    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_locals(local_bests, out_path):
    _ensure_parent_dir(out_path)
    arr = np.array(local_bests)
    plt.figure(figsize=(10, 5))
    for i in range(arr.shape[1]):
        plt.plot(arr[:, i], label=f"Wolf {i + 1}")
    plt.title("GWO Local Bests")
    plt.xlabel("Iteration")
    plt.ylabel("Personal Best Validation Accuracy")
    plt.legend(fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_curves(history, out_path_accuracy, out_path_loss):
    _ensure_parent_dir(out_path_accuracy)
    _ensure_parent_dir(out_path_loss)
    epochs = range(1, len(history["train_accuracy"]) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_accuracy"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path_accuracy)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path_loss)
    plt.close()


def plot_confusion_matrix(confusion_matrix, class_names, out_path):
    _ensure_parent_dir(out_path)
    normalized_matrix = normalize_confusion_matrix(confusion_matrix)

    plt.figure(figsize=(9, 7))
    plt.imshow(normalized_matrix, interpolation="nearest", cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
